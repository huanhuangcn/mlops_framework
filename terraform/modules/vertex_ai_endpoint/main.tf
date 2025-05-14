resource "google_vertex_ai_endpoint" "example_endpoint" {
  display_name = "example-endpoint"
  project     = var.project_id
  location    = var.region

  metadata {
    key   = "example_key"
    value = "example_value"
  }
}

output "endpoint_id" {
  value = google_vertex_ai_endpoint.example_endpoint.id
}

output "endpoint_display_name" {
  value = google_vertex_ai_endpoint.example_endpoint.display_name
}