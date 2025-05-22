terraform {
  required_providers {
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 4.61.0"
    }
  }
  required_version = ">= 1.3.0"
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

resource "google_vertex_ai_metadata_store" "default" {
  provider = google-beta
  description = "Example metadata store"
  name     = "default"
  region = var.region
  project  = var.project_id

  // Optional: Enable CMEK (Customer-Managed Encryption Key)
  // encryption_spec {
  //   kms_key_name = "projects/your-project-id/locations/your-region/keyRings/your-keyring/cryptoKeys/your-key"
  // }
}

// Optional: If you want to use CMEK, uncomment the encryption_spec block above
// and replace the kms_key_name with your KMS key.