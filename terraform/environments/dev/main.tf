terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = { // Add this for clarity and proper initialization
      source  = "hashicorp/google-beta"
      version = "~> 5.0" // Match the google provider version or use a specific one
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  credentials = file("/Users/huanhuangcn/keys/mlops-459809-052f6dbb4a52.json") // <-- ENSURE THIS PATH IS CORRECT and the SA has Vertex AI permissions
  project     = var.project_id
  region      = var.region
}

variable "project_id" {
  description = "The GCP project ID for the dev environment."
  type        = string
}

variable "region" {
  description = "The GCP region for resources in the dev environment."
  type        = string
  default     = "europe-west2"
}

variable "dev_bucket_name_suffix" {
  description = "Suffix for the dev GCS bucket name to help ensure uniqueness."
  type        = string
  default     = "dev-data-mlops"
}

module "mlops_data_bucket_dev" {
  source      = "../../modules/gcs_bucket"
  bucket_name = "${var.project_id}-${var.dev_bucket_name_suffix}"
  project_id  = var.project_id
  location    = var.region
  labels = {
    environment = "dev"
    purpose     = "mlops-data"
  }
}
