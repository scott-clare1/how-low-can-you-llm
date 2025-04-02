variable "region" {
  description = "The region in which to deploy compute."
  type        = string
}

variable "ami" {
  description = "The AMI for the machine image to create. We are defaulting to a free image."
  type        = string
  default     = "ami-000538a708e5aec0a"
}

variable "instance_type" {
  description = "The EC2 instance type to be deployed. We are defaulting to a free-tier instance."
  type        = string
  default     = "t2.micro"
}

variable "cluster_name" {

}

variable "subnet_id_a" {

}

variable "subnet_id_b" {

}
