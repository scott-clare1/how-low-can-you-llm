variable "region" {
  description = "The region in which to deploy compute."
  type        = string
}


variable "alb_name" {
  description = "The name of the ALB."
  type        = string
  default     = "alb"
}

variable "lb_target_group_name" {
  description = "The name for the load balancer target group."
  type        = string
  default     = "target-group"
}

variable "security_group_id" {}

variable "subnet_id_a" {}
variable "subnet_id_b" {

}

variable "vpc_id" {}
