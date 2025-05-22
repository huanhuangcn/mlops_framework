# Grant Vertex AI SA objectViewer rights on your models bucket
resource "google_storage_bucket_iam_member" "vertex_ai_object_viewer" {
  bucket = "mlops-459809-dev-data-mlops"
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:service-434846145915@gcp-sa-aiplatform.iam.gserviceaccount.com"
}