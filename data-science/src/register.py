
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
import json
from pathlib import Path

import mlflow


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name under which model will be registered"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the existing trained MLflow model"
    )

    parser.add_argument(
        "--model_info_output_path",
        type=str,
        required=True,
        help="Directory for model_info.json"
    )

    return parser.parse_args()


def main(args):
    """Register the existing trained MLflow model."""

    print("Registering model:", args.model_name)
    print("MLflow version:", mlflow.__version__)

    # Locate the existing model produced by the sweep
    model_path = Path(args.model_path).resolve()

    # Validate that the input is an MLflow model
    if not (model_path / "MLmodel").is_file():
        raise FileNotFoundError(
            f"MLflow model metadata not found in {model_path}"
        )

    # Register the saved model directly.
    # Do not log or serialize it again.
    model_uri = model_path.as_uri()

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=args.model_name
    )

    # Retrieve the registered model version
    model_version = registered_model.version

    print("Model registered successfully!")
    print("Model name:", args.model_name)
    print("Model version:", model_version)

    # Preserve the output expected by the pipeline
    output_dir = Path(args.model_info_output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_info = {
        "id": f"{args.model_name}:{model_version}",
        "name": args.model_name,
        "version": str(model_version)
    }

    output_path = output_dir / "model_info.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    print("Model information saved to:", output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
