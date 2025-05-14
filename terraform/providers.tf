provider "google" {
  credentials = file("<PATH_TO_YOUR_SERVICE_ACCOUNT_JSON>")
  project     = "<YOUR_PROJECT_ID>"
  region      = "<YOUR_REGION>"
}

provider "google-beta" {
  credentials = file("<PATH_TO_YOUR_SERVICE_ACCOUNT_JSON>")
  project     = "<YOUR_PROJECT_ID>"
  region      = "<YOUR_REGION>"
}