# MLOps Framework

## Overview
The MLOps Framework is designed to facilitate the lifecycle management of machine learning models using Google Cloud Platform (GCP) services, specifically Vertex AI. This framework supports multiple use cases and is built with reusability and redeployability in mind, leveraging Terraform for infrastructure as code.

## Project Structure
```
mlops_framework
├── src
│   ├── use_cases
│   │   └── example_use_case
│   │       ├── pipeline.py          # Main logic for the example use case pipeline
│   │       └── components.py        # Individual components for the pipeline
│   ├── shared_components
│   │   └── data_validation
│   │       └── component.py         # Data validation logic for reuse
│   └── utils
│       └── gcp_helpers.py           # Utility functions for GCP interactions
├── terraform
│   ├── modules
│   │   └── vertex_ai_endpoint
│   │       └── main.tf              # Terraform configuration for Vertex AI endpoint
│   ├── environments
│   │   └── dev
│   │       └── main.tf              # Terraform configuration for the development environment
│   └── providers.tf                 # Terraform providers configuration
├── config
│   └── example_use_case_config.yaml  # Configuration settings for the example use case
├── tests
│   └── test_example_use_case.py      # Unit tests for the example use case
├── requirements.txt                  # Python dependencies for the project
└── README.md                         # Project documentation
```

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd mlops_framework
   ```

2. **Install Dependencies**
   Ensure you have Python and pip installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Terraform**
   - Navigate to the `terraform` directory.
   - Update the `providers.tf` file with your GCP project settings.
   - Modify the `main.tf` files in the `modules` and `environments` directories as needed.

4. **Deploy Infrastructure**
   Use Terraform to deploy the infrastructure:
   ```bash
   cd terraform/environments/dev
   terraform init
   terraform apply
   ```

5. **Run the Example Use Case**
   Execute the pipeline script:
   ```bash
   python src/use_cases/example_use_case/pipeline.py
   ```

## Usage Examples
- To validate data, use the functions defined in `src/shared_components/data_validation/component.py`.
- For GCP interactions, utilize the helper functions in `src/utils/gcp_helpers.py`.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.