// filepath: /Users/huanhuangcn/ai/mlops_framework/terraform/modules/gcs_bucket/main.tf
resource "google_storage_bucket" "bucket" {
  name                        = var.bucket_name
  location                    = var.location
  uniform_bucket_level_access = true // Recommended for new buckets
  storage_class               = var.storage_class
  project                     = var.project_id

  versioning {
    enabled = var.versioning_enabled
  }

  labels = var.labels
}