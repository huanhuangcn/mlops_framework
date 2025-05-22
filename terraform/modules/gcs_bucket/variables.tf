// filepath: /Users/huanhuangcn/ai/mlops_framework/terraform/modules/gcs_bucket/variables.tf
variable "bucket_name" {
  description = "The name of the GCS bucket. Must be globally unique."
  type        = string
}

variable "project_id" {
  description = "The GCP project ID where the bucket will be created."
  type        = string
}

variable "location" {
  description = "The location of the GCS bucket (e.g., US-CENTRAL1, US)."
  type        = string
}

variable "storage_class" {
  description = "The storage class of the GCS bucket."
  type        = string
  default     = "STANDARD"
}

variable "versioning_enabled" {
  description = "Enable object versioning for the bucket."
  type        = bool
  default     = true
}

variable "labels" {
  description = "A map of labels to assign to the bucket."
  type        = map(string)
  default     = {}
}