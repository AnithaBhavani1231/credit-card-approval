from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from credit_card_approval.config import METRICS_PATH, MODEL_PATH
from credit_card_approval.modeling import train_best_model
from credit_card_approval.prediction import load_model, predict_application
from credit_card_approval.schemas import DEFAULT_FORM_VALUES, FORM_OPTIONS


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "credit-card-approval-local-dev"

    @app.route("/", methods=["GET", "POST"])
    def index():
        ensure_model_ready()

        form_values = dict(DEFAULT_FORM_VALUES)
        result = None
        errors: list[str] = []

        if request.method == "POST":
            form_values.update(request.form.to_dict())
            try:
                result = predict_application(form_values, get_model_payload())
            except ValueError as exc:
                errors.append(str(exc))

        return render_template(
            "index.html",
            form_values=form_values,
            form_options=FORM_OPTIONS,
            result=result,
            errors=errors,
            metrics=read_training_summary(),
        )

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        ensure_model_ready()
        payload = request.get_json(silent=True) or {}
        try:
            result = predict_application(payload, get_model_payload())
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "ok",
                "model_available": MODEL_PATH.exists(),
                "metrics_available": METRICS_PATH.exists(),
            }
        )

    return app


def ensure_model_ready() -> None:
    if MODEL_PATH.exists():
        return

    train_best_model()
    get_model_payload.cache_clear()


@lru_cache(maxsize=1)
def get_model_payload() -> dict:
    return load_model(MODEL_PATH)


def read_training_summary() -> dict:
    if not METRICS_PATH.exists():
        return {}

    import json

    with Path(METRICS_PATH).open("r", encoding="utf-8") as file:
        return json.load(file)


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
