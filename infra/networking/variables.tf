variable "region" {
  description = "The region in which to deploy compute."
  type        = string
}

variable "vpc_name" {
  description = "The name of the VPC."
  type        = string
  default     = "how-low-can-you-llm-vpc"
}

variable "vpc_cidr_block" {
  description = "The CIDR block for the VPC."
  type        = string
  default     = "172.16.0.0/16"
}

variable "subnet_name" {
  description = "The name of subnet."
  type        = string
  default     = "how-low-can-you-llm-subnet"
}

variable "subnet_cidr_block" {
  description = "The CIDR block for the subnet."
  type        = string
  default     = "172.16.10.0/24"
}

variable "availability_zone" {
  description = "The availability zone within a region to deploy the VPC, e.g., 'a' or 'b' etc."
  type        = string
  default     = "a"
}
