import os, sys

# ─ prepend the project’s src folder so absolute imports work ─
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import google.cloud.aiplatform as aiplatform
from sklearn.metrics import accuracy_score
import numpy as np
import logging
import yaml
from typing import Dict, Any

from use_cases.example_use_case.components import ExampleTrainingComponent
from utils.vertex_model_utils      import register_model_to_vertex_ai
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from google.cloud import storage, aiplatform
import os
import shutil
import datetime

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def ensure_experiment_exists(experiment_name, project, location):
    aiplatform.init(project=project, location=location)
    try:
        aiplatform.Experiment(experiment_name)
    except Exception as e:
        # If not found, create it
        aiplatform.Experiment.create(experiment_name)
        
def load_data(data_source: str) -> Any:
    """
    Load data from the specified source.
    
    Args:
        data_source (str): The source from which to load data.
    
    Returns:
        Any: Loaded data.
    """
    try:
        data = pd.read_csv(data_source)
        return data
    except Exception as e:
        logging.error(f"Error loading data from {data_source}: {e}")
        return None

def validate_data(data: Any) -> Any:
    """
    Validate the input data.
    
    Args:
        data (Any): Input data to validate.
    
    Returns:
        Any: Validated data.
    """
    if data is None:
        logging.error("Data is None. Skipping validation.")
        return None
    # Implementation for data validation
    return data

def process_data(data: Any) -> Any:
    """
    Process the validated data.
    
    Args:
        data (Any): Validated data to process.
    
    Returns:
        Any: Processed data.
    """
    if data is None:
        logging.error("Data is None. Skipping processing.")
        return None
    
    # Identify object columns
    object_cols = data.select_dtypes(include=['object']).columns
    
    # Convert object columns to numerical using Label Encoding
    for col in object_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    
    # Drop the "timestamp" column if it exists
    if "timestamp" in data.columns:
        data = data.drop(columns=["timestamp"])
    
    return data

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_model(data: Any, model_params: Dict[str, Any]) -> Any:
    """
    Train a machine learning model using the processed data.
    
    Args:
        data (Any): Processed data for training.
        model_params (Dict[str, Any]): Parameters for the model training.
    
    Returns:
        Any: Trained model.
    """
    if data is None:
        logging.error("Data is None. Skipping model training.")
        return None
    
    X = data.drop(columns=[model_params['target_column']])
    y = data[model_params['target_column']]
    
    if model_params['model_type'] == 'logistic_regression':
        model = LogisticRegression(**model_params['hyperparameters'])
    elif model_params['model_type'] == 'random_forest':
        model = RandomForestClassifier(**model_params['hyperparameters'])
    else:
        raise ValueError(f"Unsupported model type: {model_params['model_type']}")
    
    model.fit(X, y)
    return model

def deploy_model(model: Any, deployment_params: Dict[str, Any]) -> None:
    """
    Deploy the trained model to the specified endpoint.
    
    Args:
        model (Any): The trained model to deploy.
        deployment_params (Dict[str, Any]): Parameters for deployment.
    """
    # Implementation for model deployment
    pass

def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    return {"accuracy": accuracy}

def fetch_model_metrics(model_name: str, project_id: str, region: str):
    aiplatform.init(project=project_id, location=region)
    model = aiplatform.Model(model_name)
    return model.labels  # Assuming metrics are stored as labels

def log_experiment_metrics(
    experiment_name: str,
    run_name: str,
    metrics: Dict[str, float],
    params: Dict[str, Any],
):
    """
    Log metrics and parameters to Vertex AI Experiments.

    Args:
        experiment_name (str): Name of the experiment.
        run_name (str): Name of the run.
        metrics (Dict[str, float]): Metrics to log.
        params (Dict[str, Any]): Parameters to log.
    """
    aiplatform.init(project=params["project_id"], location=params["region"], experiment=experiment_name)
    with aiplatform.start_run(run=run_name) as run:
        run.log_metrics(metrics)
        run.log_params(params)
    logging.info(f"Logged metrics and parameters to Vertex AI Experiment: {experiment_name}/{run_name}")
def deploy_model_with_retry(endpoint, model, cfg, max_attempts=10, delay=30):
    for attempt in range(max_attempts):
        try:
            deployed_model = endpoint.deploy(model=model)
            logging.info(f"Model deployed to endpoint: {endpoint.name}")
            return deployed_model
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}: Error deploying model to endpoint: {e}")
            if attempt < max_attempts - 1:
                logging.info(f"Waiting {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                logging.error("Max attempts reached. Deployment failed.")
                raise e
            

def load_model_locally_from_gcs(model_artifact_gcs_dir: str, local_download_subdir: str, project_id: str, model_filename: str = 'model.joblib'):
    """
    Downloads a model artifact from GCS, loads it, and cleans up.
    Args:
        model_artifact_gcs_dir (str): GCS directory URI where the model artifact is stored.
        local_download_subdir (str): Name for the temporary local directory to download the model.
        project_id (str): GCP project ID for the storage client.
        model_filename (str): The name of the model file (e.g., 'model.joblib').
    Returns:
        Loaded model object.
    """
    logger = logging.getLogger(__name__)
    if not model_artifact_gcs_dir.startswith("gs://"):
        raise ValueError(f"Invalid GCS path for model artifact directory: {model_artifact_gcs_dir}")

    gcs_full_path_to_model_file = f"{model_artifact_gcs_dir.rstrip('/')}/{model_filename}"
    
    # Ensure the base directory for temporary downloads exists
    base_temp_dir = "./temp_model_downloads" # Or any other appropriate base path
    os.makedirs(base_temp_dir, exist_ok=True)
    specific_local_model_download_dir = os.path.join(base_temp_dir, local_download_subdir)
    
    os.makedirs(specific_local_model_download_dir, exist_ok=True)
    local_model_file_path = os.path.join(specific_local_model_download_dir, model_filename)

    try:
        logger.info(f"Downloading model artifact from {gcs_full_path_to_model_file} to {local_model_file_path}")
        
        storage_client = storage.Client(project=project_id)
        bucket_name, blob_name = gcs_full_path_to_model_file.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_model_file_path)
        
        logger.info("Model artifact downloaded. Loading model.")
        loaded_model = joblib.load(local_model_file_path)
        return loaded_model
    finally:
        if os.path.exists(specific_local_model_download_dir):
            logger.info(f"Cleaning up temporary download directory: {specific_local_model_download_dir}")
            shutil.rmtree(specific_local_model_download_dir)


def compare_models(vertex_model, existing_model_name_from_config, validation_data, cfg):
    """Compares the new model against an existing model (if provided) using local predictions for both."""
    logger = logging.getLogger(__name__)
    X_val, y_val = validation_data # Assuming X_val is a DataFrame and y_val is a Series/array

    new_model_metrics = {}
    existing_model_metrics = {}
    model_filename = 'model.joblib' # Standard model filename

    # --- New model prediction (load locally) ---
    try:
        logger.info(f"Attempting to load new model (version: {vertex_model.version_id}) locally from artifact URI: {vertex_model.uri}")
        loaded_new_model = load_model_locally_from_gcs(
            model_artifact_gcs_dir=vertex_model.uri,
            local_download_subdir="temp_downloaded_new_model",
            project_id=cfg.get("project_id"),
            model_filename=model_filename
        )
        logger.info("New model loaded locally. Performing predictions.")
        new_predictions_array = loaded_new_model.predict(X_val) 
        new_accuracy = accuracy_score(y_val, np.round(new_predictions_array))
        new_model_metrics = {'accuracy': new_accuracy}
        logger.info(f"New model accuracy (local prediction): {new_accuracy}")

    except Exception as e_new_model_predict:
        logger.error(f"Error during new model local prediction: {e_new_model_predict}", exc_info=True)
        logger.warning("Skipping new model metrics calculation due to error.")

    # --- Existing model prediction (load locally) ---
    if existing_model_name_from_config: # Check if existing_model_name is in config
        logger.info(f"Existing model specified in config: {existing_model_name_from_config}")
        try:
            # Fetch the existing model details from Vertex AI to get its artifact URI
            # Ensure aiplatform is initialized if not done globally for this specific call context
            aiplatform.init(project=cfg.get("project_id"), location=cfg.get("region"))
            existing_vertex_model_resource = aiplatform.Model(existing_model_name_from_config)
            existing_model_artifact_uri = existing_vertex_model_resource.uri
            
            if not existing_model_artifact_uri:
                raise ValueError(f"Artifact URI not found for existing model: {existing_model_name_from_config}")

            logger.info(f"Attempting to load existing model (name: {existing_model_name_from_config}, version: {existing_vertex_model_resource.version_id}) locally from artifact URI: {existing_model_artifact_uri}")
            loaded_existing_model = load_model_locally_from_gcs(
                model_artifact_gcs_dir=existing_model_artifact_uri,
                local_download_subdir="temp_downloaded_existing_model",
                project_id=cfg.get("project_id"), # Assuming existing model's GCS bucket is accessible by this project
                model_filename=model_filename
            )
            logger.info("Existing model loaded locally. Performing predictions.")
            existing_predictions_array = loaded_existing_model.predict(X_val) # X_val is DataFrame
            existing_accuracy = accuracy_score(y_val, np.round(existing_predictions_array))
            existing_model_metrics = {'accuracy': existing_accuracy}
            logger.info(f"Existing model accuracy (local prediction): {existing_accuracy}")

        except Exception as e_existing_local_predict:
            logger.error(f"Error during existing model local prediction: {e_existing_local_predict}", exc_info=True)
            logger.warning("Skipping existing model metrics calculation due to error.")
    else:
        logger.info("No existing model name provided in config. Skipping comparison with existing model.")

    return new_model_metrics, existing_model_metrics

def main(config_path: str):
    """
    Main entry for the MLOps pipeline PoC.

    Steps:
    1. Load pipeline configuration from YAML.
    2. Ensure the Vertex AI experiment exists (for tracking runs).
    3. Train a new model and register it to Vertex AI Model Registry.
    4. Prepare and process the input data.
    5. Compare the new model with an existing model (if provided in config).
    6. Log the comparison results to Vertex AI Experiments.
    """
    # 1. Load configuration
    cfg = load_config(config_path)
    # 2. Ensure experiment exists for tracking
    ensure_experiment_exists_for_cfg(cfg)
    # 3. Train and register model
    model_uri, vertex_model = train_and_register(cfg)
    # 4. Prepare and process data
    processed_data = prepare_data(cfg)
    # 5. Compare new model with existing model (if any)
    new_metrics, existing_metrics = compare_with_existing(vertex_model, processed_data, cfg)
    # 6. Log comparison results
    log_comparison(vertex_model, new_metrics, existing_metrics, cfg)
    # Optionally: deploy_model_to_endpoint(vertex_model, cfg)

# --- Helper functions for clarity ---

def load_config(config_path):
    logging.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)

def ensure_experiment_exists_for_cfg(cfg):
    ensure_experiment_exists(
        experiment_name="cat-tastrophe-model-comparison",
        project=cfg["project_id"],
        location=cfg["region"]
    )

def train_and_register(cfg):
    trainer = ExampleTrainingComponent(cfg)
    model_uri = trainer.execute()
    if not model_uri:
        raise RuntimeError("Training failed, no artifact URI returned")
    vertex_model = trainer.register_model_to_vertex_ai(
        model_artifact_uri=model_uri,
        display_name=cfg["model_display_name"],
        description=cfg.get("model_description", ""),
        serving_container_image_uri=cfg["serving_container_image_uri"],
        labels=cfg["labels"]
    )
    return model_uri, vertex_model

def prepare_data(cfg):
    data = load_data(cfg["data_source"])
    validated = validate_data(data)
    if validated is None:
        raise RuntimeError("Invalid validated data.")
    processed = process_data(validated)
    if processed is None:
        raise RuntimeError("Invalid processed data.")
    return processed

def compare_with_existing(vertex_model, processed_data, cfg):
    X_val = processed_data.drop(columns=[cfg['training_params']['target_column']])
    y_val = processed_data[cfg['training_params']['target_column']]
    return compare_models(vertex_model, cfg.get("existing_model_name"), (X_val, y_val), cfg)

def log_comparison(vertex_model, new_metrics, existing_metrics, cfg):
    log_model_comparison_to_vertex_ai_experiment(
        vertex_model, new_metrics, existing_metrics, cfg
    )
    
def log_model_comparison_to_vertex_ai_experiment(
    vertex_model, new_model_metrics, existing_model_metrics, cfg
):
    experiment_name = "cat-tastrophe-model-comparison"
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Log new model metrics
    if new_model_metrics:
        run_name = f"new-model-{vertex_model.version_id}-{timestamp}"
        log_experiment_metrics(
            experiment_name=experiment_name,
            run_name=run_name,
            metrics=new_model_metrics,
            params={
                "project_id": cfg["project_id"],
                "region": cfg["region"],
                "model_version": vertex_model.version_id,
                "model_type": cfg["training_params"]["model_type"],
                "run_type": "new"
            }
        )

    # Log existing model metrics
    if existing_model_metrics and cfg.get("existing_model_name"):
        existing_model_id = cfg["existing_model_name"].split("/")[-1]
        run_name = f"existing-model-{existing_model_id}-{timestamp}"
        log_experiment_metrics(
            experiment_name=experiment_name,
            run_name=run_name,
            metrics=existing_model_metrics,
            params={
                "project_id": cfg["project_id"],
                "region": cfg["region"],
                "model_version": existing_model_id,
                "model_type": "unknown",
                "run_type": "existing"
            }
        )
    logging.info(f"Logged model comparison to Vertex AI Experiment: {experiment_name}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])








