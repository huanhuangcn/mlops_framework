variable "project_id" {
  description = "The GCP project ID."
  type        = string
}

variable "region" {
  description = "The GCP region for resources."
  type        = string
  default     = "europe-west2"
}


provider "google" {
  project = "mlops-459809"
  region  = "europe-west2"
}

# Data source to get the project
data "google_project" "project" {
  project_id = "mlops-459809"
}

# Data source to get the service account
data "google_service_account" "default" {
  account_id = "terraform-mlops-admin"
  project    = "mlops-459809"
}

# Grant the aiplatform.modelUser role to the service account
resource "google_project_iam_member" "aiplatform_model_user" {
  project = "mlops-459809"
  role    = "roles/aiplatform.modelUser"
  member  = "serviceAccount:${data.google_service_account.default.email}"
}

# Grant the storage.objectViewer role to the service account
resource "google_project_iam_member" "storage_object_viewer" {
  project = "mlops-459809"
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${data.google_service_account.default.email}"
}

# Grant the aiplatform.admin role to the service account
resource "google_project_iam_member" "aiplatform_admin" {
  project = "mlops-459809"
  role    = "roles/aiplatform.admin"
  member  = "serviceAccount:${data.google_service_account.default.email}"
}