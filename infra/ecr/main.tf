provider "aws" {
  region = "eu-west-2"
}


resource "aws_ecr_repository" "how-low-can-you-llm-ecr" {
  name = var.ecr_name

  image_scanning_configuration {
    scan_on_push = true
  }
}
