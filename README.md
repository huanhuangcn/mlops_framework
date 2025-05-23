# MLOps Framework PoC

This repository demonstrates a practical MLOps pipeline using Google Vertex AI, scikit-learn, and other modern ML tools. The pipeline covers the full ML lifecycle: data loading, validation, processing, model training, registration, comparison, and experiment tracking.

## Features

- **Config-driven pipeline**: Easily customize runs via YAML config files.
- **Data validation & preprocessing**: Ensures clean, ready-to-train data.
- **Model training**: Supports scikit-learn models (RandomForest, LogisticRegression, etc.).
- **Model registration**: Registers models to Vertex AI Model Registry.
- **Model comparison**: Compares new and existing models using local predictions.
- **Experiment tracking**: Logs metrics and parameters to Vertex AI Experiments.
- **Deployment utilities**: Includes helpers for model deployment and retry logic.

## Getting Started

### 1. Clone the repository

```sh
git clone https://github.com/your-org/mlops_framework.git
cd mlops_framework
```

### 2. Create and activate the conda environment

```sh
conda env create -f environment.yml
conda activate mlops_env
```

### 3. Prepare your configuration

Edit or create a YAML config file (see `example_config.yaml`) with your project, region, data source, and model parameters.

### 4. Run the pipeline

```sh
python src/use_cases/example_use_case/pipeline.py /path/to/your/config.yaml
```

## Project Structure

```
mlops_framework/
├── environment.yml
├── src/
│   ├── use_cases/
│   │   └── example_use_case/
│   │       ├── pipeline.py
│   │       └── components.py
│   └── utils/
│       └── vertex_model_utils.py
└── ...
```

## Key Files

- **pipeline.py**: Main pipeline logic and orchestration.
- **components.py**: Training component implementation.
- **vertex_model_utils.py**: Utilities for Vertex AI model operations.

## Configuration Example

```yaml
project_id: "your-gcp-project"
region: "us-central1"
data_source: "data/train.csv"
model_display_name: "cat-tastrophe-model"
serving_container_image_uri: "gcr.io/your-project/your-serving-image"
labels:
  team: "ml"
  env: "dev"
training_params:
  model_type: "random_forest"
  target_column: "label"
  hyperparameters:
    n_estimators: 100
    max_depth: 5
existing_model_name: "projects/your-gcp-project/locations/us-central1/models/1234567890"
```

## Tips

- **VS Code**: You can run and debug the pipeline directly in VS Code. Make sure to pass the config path as an argument.
- **Google Cloud**: Ensure your environment is authenticated (`gcloud auth application-default login`) and has the necessary permissions.
- **Extending**: Add new models, metrics, or steps by editing `components.py` and `pipeline.py`.

## Troubleshooting

- If you see `Usage: python pipeline.py <config_path>`, provide the path to your config file as an argument.
- For missing dependencies, update your environment with `conda env update -f environment.yml`.
- For Google Cloud errors, check your credentials and project/region settings.

## License

MIT License

---

*For questions or contributions, please open an issue or pull request.*