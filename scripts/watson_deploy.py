from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from credit_card_approval.config import MODEL_PATH


def main() -> None:
    """Deploy the saved model package to IBM Watson Machine Learning.

    Required environment variables:
    - IBM_CLOUD_API_KEY
    - IBM_WML_URL
    - IBM_WML_SPACE_ID
    - IBM_WML_SOFTWARE_SPEC
    - IBM_WML_MODEL_TYPE

    This script is intentionally small and explicit so credentials never appear in
    source code. Install the optional package first:
    pip install -r requirements-watson.txt
    """
    try:
        from ibm_watson_machine_learning import APIClient
    except ImportError as exc:
        raise SystemExit("Install IBM Watson dependencies with: pip install -r requirements-watson.txt") from exc

    api_key = os.getenv("IBM_CLOUD_API_KEY")
    service_url = os.getenv("IBM_WML_URL")
    space_id = os.getenv("IBM_WML_SPACE_ID")
    software_spec_name = os.getenv("IBM_WML_SOFTWARE_SPEC")
    model_type = os.getenv("IBM_WML_MODEL_TYPE")

    if not all([api_key, service_url, space_id, software_spec_name, model_type]):
        raise SystemExit(
            "Set IBM_CLOUD_API_KEY, IBM_WML_URL, IBM_WML_SPACE_ID, "
            "IBM_WML_SOFTWARE_SPEC, and IBM_WML_MODEL_TYPE before running deployment."
        )

    if not MODEL_PATH.exists():
        raise SystemExit(f"Train the model first. Missing file: {MODEL_PATH}")

    client = APIClient({"url": service_url, "apikey": api_key})
    client.set.default_space(space_id)

    software_spec_uid = client.software_specifications.get_uid_by_name(software_spec_name)
    if not software_spec_uid:
        raise SystemExit(f"IBM Watson software specification not found: {software_spec_name}")

    metadata = {
        client.repository.ModelMetaNames.NAME: "Credit Card Approval Prediction Model",
        client.repository.ModelMetaNames.TYPE: model_type,
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid,
    }

    model_details = client.repository.store_model(model=str(Path(MODEL_PATH).resolve()), meta_props=metadata)
    model_uid = client.repository.get_model_id(model_details)

    deployment_details = client.deployments.create(
        artifact_uid=model_uid,
        meta_props={
            client.deployments.ConfigurationMetaNames.NAME: "Credit Card Approval API",
            client.deployments.ConfigurationMetaNames.ONLINE: {},
        },
    )
    print("Model UID:", model_uid)
    print("Deployment UID:", client.deployments.get_uid(deployment_details))


if __name__ == "__main__":
    main()
