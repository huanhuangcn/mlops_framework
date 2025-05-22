// filepath: /Users/huanhuangcn/ai/mlops_framework/terraform/modules/gcs_bucket/outputs.tf
output "bucket_name" {
  description = "The name of the created GCS bucket."
  value       = google_storage_bucket.bucket.name
}

output "bucket_url" {
  description = "The self_link of the GCS bucket."
  value       = google_storage_bucket.bucket.self_link
}