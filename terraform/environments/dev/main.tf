resource "google_project" "mlops_project" {
  name       = "mlops_project"
  project_id = "mlops-project-id"
  org_id     = "your-org-id"
  billing_account = "your-billing-account-id"
}

resource "google_vertex_ai_endpoint" "example_endpoint" {
  display_name = "example-endpoint"
  project      = google_project.mlops_project.project_id
  location     = "us-central1"
}

resource "google_vertex_ai_model" "example_model" {
  display_name = "example-model"
  project      = google_project.mlops_project.project_id
  location     = "us-central1"
  artifact_uri = "gs://your-bucket/path/to/model"
  serving_container {
    image_uri = "gcr.io/your-project-id/your-model-image"
  }
}

resource "google_vertex_ai_endpoint_deployment" "example_deployment" {
  endpoint = google_vertex_ai_endpoint.example_endpoint.id
  model    = google_vertex_ai_model.example_model.id
  display_name = "example-deployment"
  traffic_split = {
    "0" = 100
  }
}