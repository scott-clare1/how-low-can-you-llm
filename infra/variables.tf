variable "region" {
  description = "The region in which to deploy compute."
  type        = string
}

variable "ecr_name" {
  description = "The name of the container registry."
  type        = string
}

variable "aws_account_id" {
  description = "The AWS account ID."
  type        = string
}

variable "image_name" {
  description = "The name of the image to pull from ECR."
  type        = string
}

variable "ecs_cluster_name" {
  description = "The name of the ECS cluster."
  type        = string
}
