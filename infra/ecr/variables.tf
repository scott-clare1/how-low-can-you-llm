variable "ecr_name" {
  description = "The name of the container registry."
  type        = string
}

variable "region" {
  description = "The region in which to deploy the registry."
  type        = string
}
