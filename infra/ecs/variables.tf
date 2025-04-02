variable "region" {
  description = "The region in which to deploy compute."
  type        = string
}

variable "ecs_service_name" {
  description = "The name of the service."
  type        = string
  default     = "how-low-can-you-llm-ecs-name"
}

variable "ecs_cluster_name" {
  description = "The name of the ECS cluster."
  type        = string
}

variable "ecs_task_name" {
  description = "The name of the ECS task."
  type        = string
  default     = "how-low-can-you-llm-ecs-task"
}

variable "aws_account_id" {
  description = "The AWS account ID."
  type        = string
}

variable "container_name" {
  description = "The name of the container to be deployed."
  type        = string
  default     = "server"
}

variable "image_name" {
  description = "The name of the image to pull from ECR."
  type        = string
}

variable "ecr_name" {
  description = "The name of the container registry."
  type        = string
}

variable "subnet_id_a" {}
variable "subnet_id_b" {

}

variable "security_group_id" {
}


variable "autoscaler_arn" {}

variable "load_balancer_target_group_arn" {}
