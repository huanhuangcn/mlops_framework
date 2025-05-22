import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from google.cloud import storage
from google.cloud import aiplatform
import logging
import os
import time
import yaml  # For loading config in example usage
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier  # Or your model type
import torch
import torch.nn as nn

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class ExampleTrainingComponent:
    def __init__(self, config: dict):
        """
        Initializes the component with configuration.
        Args:
            config (dict): Configuration dictionary containing:
              - data_source (str): GCS CSV path.
              - project_id (str)
              - region (str)
              - model_gcs_path_prefix (str): GCS prefix for saving models.
              - training_params (dict)
        """
        self.config = config
        logger.info("Initializing ExampleTrainingComponent...")
        logger.info(f"Project ID: {config.get('project_id')}")
        logger.info(f"Region: {config.get('region')}")
        logger.info(f"Data Source: {config.get('data_source')}")
        logger.info(f"Model GCS Path Prefix: {config.get('model_gcs_path_prefix')}")
        logger.info(f"Training Parameters: {config.get('training_params')}")

        try:
            self.storage_client = storage.Client(project=config.get('project_id'))
            logger.info("GCS client initialized.")
        except Exception as e:
            logger.error("Failed to init GCS client", exc_info=True)
            raise

    def _upload_to_gcs(self, local_path: str, bucket_name: str, blob_name: str) -> str:
        logger.info(f"Uploading {local_path} → gs://{bucket_name}/{blob_name}")
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        uri = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Uploaded to {uri}")
        return uri

    def load_data(self) -> pd.DataFrame:
        path = self.config['data_source']
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded DataFrame shape: {df.shape}")
        logger.debug(f"DataFrame head:\n{df.head()}")
        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Processing data (convert categorical to numerical)...")
        if df.empty:
            logger.warning("Empty DataFrame, skipping processing.")
            return df

        # Remove timestamp column if present
        if "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])

        # Convert all object/category columns to numerical using LabelEncoder
        for col in df.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            logger.info(f"Encoded column '{col}' with LabelEncoder.")

        logger.info(f"Columns after encoding: {df.columns.tolist()}")
        return df

    def train_model(self, df: pd.DataFrame) -> str:
        logger.info("Training model...")
        tp = self.config.get('training_params', {})
        target = tp.get('target_column', 'target')
        X = df.drop(columns=[target])
        y = df[target]

        vs = tp.get('validation_split', 0.2)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=vs, random_state=42)
        logger.info(f"Train shape {X_train.shape}, Val shape {X_val.shape}")

        params = {
            'solver': tp.get('solver', 'liblinear'),
            'C': tp.get('C', 1.0),
            'random_state': tp.get('random_state', 42)
        }
        logger.info(f"Model params: {params}")
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        logger.info(f"Validation accuracy: {acc:.4f}")

        fname = "model.joblib"
        joblib.dump(model, fname)
        logger.info(f"Saved local model to {fname}")

        prefix = self.config['model_gcs_path_prefix']
        if not prefix.startswith("gs://"):
            raise ValueError("model_gcs_path_prefix must start with gs://")
        bucket = prefix.split('/')[2]
        path = "/".join(prefix.split('/')[3:]).strip('/')
        ts = time.strftime("%Y%m%d%H%M%S")
        blob = f"{path}/{ts}/{fname}"
        uri = self._upload_to_gcs(fname, bucket, blob)
        #os.remove(fname)
        logger.info(f"Removed local file {fname}")

        directory_uri = os.path.dirname(uri) + "/"
        logger.info(f"Model artifact directory: {directory_uri}")
        return directory_uri

    def sklearn_to_pytorch(self, sklearn_model, input_dim):
        pytorch_model = LogisticRegressionModel(input_dim)
        # Extract weights and bias from sklearn model
        weights = sklearn_model.coef_[0]
        bias = sklearn_model.intercept_[0]

        # Set weights and bias to pytorch model
        pytorch_model.linear.weight.data = torch.tensor(weights, dtype=torch.float32).reshape(1, -1)
        pytorch_model.linear.bias.data = torch.tensor(bias, dtype=torch.float32)
        return pytorch_model

    def register_model_to_vertex_ai(
        self,
        model_artifact_uri: str,
        display_name: str,
        serving_container_image_uri: str,
        description: str = "",
        labels: dict = None,
    ):
        """
        Registers the trained model artifact to Vertex AI Model Registry.

        Args:
            model_artifact_uri (str): GCS URI to the model directory (should end with '/').
            display_name (str): Display name for the model in Vertex AI.
            serving_container_image_uri (str): URI of the serving container image.
            description (str): Model description.
            labels (dict): Optional labels for the model.
        Returns:
            aiplatform.Model: The registered Vertex AI model object.
        """
        project_id = self.config["project_id"]
        region = self.config["region"]

        aiplatform.init(project=project_id, location=region)
        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=model_artifact_uri,
            serving_container_image_uri=serving_container_image_uri,
            description=description,
            labels=labels or {},
            sync=True,
        )
        logging.info(f"Registered model to Vertex AI: {model.resource_name}")
        return model

    def execute(self) -> str:
        logger.info("Executing ExampleTrainingComponent...")
        start = time.time()
        df = self.load_data()
        if df.empty:
            logger.error("No data loaded, abort.")
            return None

        dfp = self.process_data(df)
        if dfp.empty:
            logger.error("Processing yielded empty DataFrame, abort.")
            return None

        model_uri = self.train_model(dfp)

        # Load the trained scikit-learn model
        model = joblib.load("model.joblib")  # Load the model

        # Convert sklearn model to pytorch
        input_dim = dfp.shape[1] - 1
        pytorch_model = self.sklearn_to_pytorch(model, input_dim)

        # Save the PyTorch model
        try:
            local_model_path = "pytorch_model.pth"
            torch.save(pytorch_model.state_dict(), local_model_path)
            logger.info(f"PyTorch model saved to local file: {local_model_path}")

            # Upload the local file to GCS
            bucket_name = model_uri.split('/')[2]
            blob_path = '/'.join(model_uri.split('/')[3:]) + "pytorch_model.pth"
            self._upload_to_gcs(local_model_path, bucket_name, blob_path)
            logger.info(f"Model uploaded to GCS: {model_uri}")

            # Remove the local temporary file
            os.remove(local_model_path)
            logger.info(f"Removed local file: {local_model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None

        duration = time.time() - start
        logger.info(f"Execution completed in {duration:.1f}s, model at {model_uri}")
        return model_uri


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


if __name__ == '__main__':
    logger.info("Running ExampleTrainingComponent as script...")
    # Prepare a real config for local testing
    config = {
        'data_source':            'gs://mlops-459809-dev-data-mlops/data/cosmic_cat_data.csv',
        'project_id':             'mlops-459809',
        'region':                 'europe-west2',
        'model_gcs_path_prefix':  'gs://mlops-459809-dev-data-mlops/models/CosmicCatTest/',
        'training_params': {
            'validation_split':   0.2,
            'target_column':      'interference_occurred',
            # 'feature_columns':  ['Feature1','Feature2'],  # optional
            'solver':             'liblinear',
            'C':                  1.0,
            'random_state':       42
        }
    }

    # Write temp config
    cfg_path = 'temp_config.yaml'
    with open(cfg_path, 'w') as fp:
        yaml.dump(config, fp)
    logger.info(f"Wrote temp config to {cfg_path}, please edit only if needed.")

    # Load and run
    with open(cfg_path) as fp:
        cfg = yaml.safe_load(fp)
    comp = ExampleTrainingComponent(cfg)
    comp.execute()