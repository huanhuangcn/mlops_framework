from google.cloud import aiplatform
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_model_to_vertex_ai(
    project_id: str,
    region: str,
    model_display_name: str,
    serving_container_image_uri: str,
    model_artifact_uri: str, # GCS URI to the directory containing your saved model
    model_description: str = "No description provided.",
    is_default_version: bool = True,
    parent_model_resource_name: str = None, # Optional: if you want to version an existing model
    labels: dict = None
):
    """
    Registers a trained model to Vertex AI Model Registry.
    If a model with model_display_name exists, it creates a new version.
    Otherwise, it creates a new model resource.

    Args:
        project_id: Your GCP project ID.
        region: The GCP region for Vertex AI.
        model_display_name: The display name for the model in Vertex AI.
        serving_container_image_uri: URI of the serving container image.
        model_artifact_uri: GCS URI to the directory containing your saved model artifacts
                            (e.g., "gs://your-bucket/path/to/model/").
        model_description: A description for the model.
        is_default_version: Whether this version should be the default.
        parent_model_resource_name: Optional. The resource name of an existing model to upload a new version to.
                                      Format: "projects/{project}/locations/{location}/models/{model_id}"
        labels: Dictionary of labels to associate with the model version.

    Returns:
        The uploaded model resource object from Vertex AI.
    """
    aiplatform.init(project=project_id, location=region)

    model_labels = labels if labels else {}

    # Check if a model with this display name already exists to get its resource name
    # This is one way to handle versioning under an existing model entry.
    # Alternatively, if parent_model_resource_name is provided, use that directly.
    if not parent_model_resource_name:
        existing_models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
        if existing_models:
            parent_model_resource_name = existing_models[0].resource_name
            logger.info(f"Found existing model '{model_display_name}'. Will upload a new version to: {parent_model_resource_name}")
        else:
            logger.info(f"No existing model found with display name '{model_display_name}'. A new model entry will be created.")


    try:
        logger.info(f"Starting model upload for '{model_display_name}'...")
        logger.info(f"  Artifact URI: {model_artifact_uri}")
        logger.info(f"  Serving Container: {serving_container_image_uri}")

        # The upload_model method handles both creating a new model and versioning an existing one
        # if parent_model is specified.
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_artifact_uri,
            serving_container_image_uri=serving_container_image_uri,
            serving_container_predict_route="/predict",  # Adjust if your container uses a different route
            serving_container_health_route="/health",    # Adjust if your container uses a different route
            description=model_description,
            parent_model=parent_model_resource_name, # Key for versioning an existing model
            is_default_version=is_default_version,
            labels=model_labels,
            # sync=True # Set to False for asynchronous upload, True to wait for completion
        )

        model.wait() # Wait for the model upload to complete

        logger.info(f"Model uploaded successfully: {model.resource_name}")
        logger.info(f"  Display Name: {model.display_name}")
        logger.info(f"  Version ID: {model.version_id}")
        logger.info(f"  Version Aliases: {model.version_aliases}")
        logger.info(f"  Resource name: {model.resource_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to upload model '{model_display_name}': {e}")
        raise

if __name__ == '__main__':
    # --- Configuration (Example - fetch from your config files or pipeline parameters) ---
    PROJECT_ID = "mlops-459809"  # Replace with your project ID
    REGION = "europe-west2"    # Replace with your region

    # Model details from your use case config or training output
    MODEL_DISPLAY_NAME = "CosmicCatTastropheForecaster" # From example_use_case_config.yaml
    MODEL_DESCRIPTION = "Predicts the probability of feline interference during astronomical observations. v2" # From example_use_case_config.yaml
    SERVING_CONTAINER_IMAGE_URI = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest" # From example_use_case_config.yaml

    # This should point to the GCS directory where your *trained model artifacts* are stored.
    # For example, if your training job saves the model to gs://<bucket>/models/cosmic_cat_model/YYYYMMDDHHMMSS/
    # This needs to be a directory containing the saved_model.pb or equivalent model files.
    # Ensure this directory structure is what your serving container expects.
    # For TensorFlow SavedModel, this is the parent directory of 'saved_model.pb' and 'variables/'.
    # For scikit-learn, this might be the directory containing 'model.joblib'.
    timestamp = time.strftime("%Y%m%d%H%M%S")
    MODEL_ARTIFACT_URI = f"gs://{PROJECT_ID}-dev-data-mlops/models/{MODEL_DISPLAY_NAME.lower().replace(' ', '_')}/{timestamp}/"
    
    # Example: Create a dummy model artifact directory for testing this script
    # In a real scenario, your training process would create this.
    from google.cloud import storage as gcs_storage
    try:
        storage_client = gcs_storage.Client(project=PROJECT_ID)
        bucket_name_only = f"{PROJECT_ID}-dev-data-mlops"
        blob_path_dummy_file = f"models/{MODEL_DISPLAY_NAME.lower().replace(' ', '_')}/{timestamp}/dummy_model_file.txt"
        bucket = storage_client.bucket(bucket_name_only)
        blob = bucket.blob(blob_path_dummy_file)
        blob.upload_from_string("This is a dummy model artifact.", content_type="text/plain")
        logger.info(f"Uploaded dummy model artifact to {MODEL_ARTIFACT_URI}dummy_model_file.txt for testing.")
    except Exception as e:
        logger.warning(f"Could not create dummy model artifact for testing: {e}. Ensure MODEL_ARTIFACT_URI points to a valid model.")
    # --- End Configuration ---

    custom_labels = {
        "framework_version": "0.1.0",
        "trained_by": "mlops_pipeline_example",
        "dataset_version": "20250515"
    }

    try:
        uploaded_model = register_model_to_vertex_ai(
            project_id=PROJECT_ID,
            region=REGION,
            model_display_name=MODEL_DISPLAY_NAME,
            serving_container_image_uri=SERVING_CONTAINER_IMAGE_URI,
            model_artifact_uri=MODEL_ARTIFACT_URI, # This is the GCS *directory*
            model_description=MODEL_DESCRIPTION,
            labels=custom_labels
        )
        if uploaded_model:
            print(f"\nSuccessfully registered model: {uploaded_model.name}")
            print(f"View it in the Vertex AI Model Registry: "
                  f"https://console.cloud.google.com/vertex-ai/models?project={PROJECT_ID}")

    except Exception as e:
        print(f"An error occurred: {e}")