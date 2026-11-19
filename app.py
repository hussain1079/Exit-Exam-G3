from functools import lru_cache
import math
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_revenue_prediction_model.pkl"

NUMERIC_FEATURES = (
    "employees",
    "employees_year",
    "net_income",
    "market_cap",
    "legal_units_count",
    "direct_subsidiaries_count",
    "max_hierarchy_depth",
)
BOOLEAN_FEATURE = "is_standalone_entity"
CATEGORY_FIELDS = {
    "jurisdiction": "enterprise_group_jurisdiction_",
    "company_type": "company_type_",
}


class InputError(ValueError):
    pass


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Prediction model not found: {MODEL_PATH}")

    with MODEL_PATH.open("rb") as model_file:
        artifact = pickle.load(model_file)

    model = artifact.get("model") if isinstance(artifact, dict) else artifact
    if model is None:
        raise RuntimeError("The model file does not contain a prediction model.")

    feature_names = tuple(str(name) for name in getattr(model, "feature_names_in_", ()))
    required_features = set(NUMERIC_FEATURES) | {BOOLEAN_FEATURE}
    if not feature_names or not required_features.issubset(feature_names):
        raise RuntimeError("The model does not expose the expected revenue prediction features.")

    recognized_features = required_features | {
        feature
        for prefix in CATEGORY_FIELDS.values()
        for feature in feature_names
        if feature.startswith(prefix)
    }
    if set(feature_names) != recognized_features:
        raise RuntimeError("The model contains features this application cannot provide.")

    categories = {
        field: tuple(
            feature[len(prefix):]
            for feature in feature_names
            if feature.startswith(prefix)
        )
        for field, prefix in CATEGORY_FIELDS.items()
    }
    if any(not values for values in categories.values()):
        raise RuntimeError("The model is missing a supported category list.")

    model._revenue_app_schema = {
        "feature_names": feature_names,
        "categories": categories,
    }
    return model


def build_feature_row(payload, schema):
    if not isinstance(payload, dict):
        raise InputError("Provide the prediction inputs as an object.")

    expected_fields = set(NUMERIC_FEATURES) | {BOOLEAN_FEATURE} | set(CATEGORY_FIELDS)
    unexpected_fields = set(payload) - expected_fields
    if unexpected_fields:
        raise InputError(f"Unexpected input field(s): {', '.join(sorted(unexpected_fields))}.")

    category_values = {}
    for field, options in schema["categories"].items():
        value = payload.get(field)
        if not isinstance(value, str) or value not in options:
            raise InputError(f"Choose a supported value for {field.replace('_', ' ')}.")
        category_values[field] = value

    numeric_values = {}
    for field in NUMERIC_FEATURES:
        raw_value = payload.get(field)
        if raw_value is None or raw_value == "":
            raise InputError(f"{field.replace('_', ' ').capitalize()} is required.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise InputError(f"{field.replace('_', ' ').capitalize()} must be a number.") from None
        if not math.isfinite(value):
            raise InputError(f"{field.replace('_', ' ').capitalize()} must be a finite number.")
        numeric_values[field] = value

    raw_boolean = payload.get(BOOLEAN_FEATURE)
    if isinstance(raw_boolean, bool):
        boolean_value = int(raw_boolean)
    elif str(raw_boolean).strip().lower() in {"1", "true", "yes"}:
        boolean_value = 1
    elif str(raw_boolean).strip().lower() in {"0", "false", "no"}:
        boolean_value = 0
    else:
        raise InputError("Choose yes or no for standalone entity.")

    row = {}
    for feature in schema["feature_names"]:
        if feature in numeric_values:
            row[feature] = numeric_values[feature]
        elif feature == BOOLEAN_FEATURE:
            row[feature] = boolean_value
        else:
            for field, prefix in CATEGORY_FIELDS.items():
                if feature.startswith(prefix):
                    row[feature] = int(feature == f"{prefix}{category_values[field]}")
                    break
    return row


def predict_revenue(payload):
    model = get_model()
    schema = model._revenue_app_schema
    row = build_feature_row(payload, schema)
    features = pd.DataFrame([row], columns=schema["feature_names"])
    prediction = float(model.predict(features)[0])
    if not math.isfinite(prediction):
        raise RuntimeError("The model returned a non-finite revenue prediction.")
    return prediction


def get_form_schema():
    schema = get_model()._revenue_app_schema
    return {
        "jurisdictions": schema["categories"]["jurisdiction"],
        "company_types": schema["categories"]["company_type"],
        "numeric_features": NUMERIC_FEATURES,
    }


@app.get("/")
def index():
    return render_template("home.html")


@app.get("/predict")
def prediction_page():
    return render_template("index.html", schema=get_form_schema(), values={})


@app.post("/predict")
def predict_form():
    values = request.form.to_dict()
    try:
        prediction = predict_revenue(values)
    except InputError as error:
        return render_template(
            "index.html",
            schema=get_form_schema(),
            values=values,
            error=str(error),
        ), 400

    return render_template(
        "index.html",
        schema=get_form_schema(),
        values=values,
        prediction=prediction,
    )


@app.post("/api/predict")
def predict_api():
    if not request.is_json:
        return jsonify(error="Send the request body as application/json."), 400
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a valid JSON object."), 400
    try:
        prediction = predict_revenue(payload)
    except InputError as error:
        return jsonify(error=str(error)), 400
    return jsonify(predicted_revenue=prediction, currency="USD")


@app.get("/health")
def health():
    get_model()
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run()
