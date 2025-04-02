provider "aws" {
  region = var.region
}

resource "aws_ecr_repository" "how-low-can-you-llm-ecr" {
  name = var.ecr_name

  image_scanning_configuration {
    scan_on_push = true
  }

  provisioner "local-exec" {
    command = "./../scripts/build-and-push.sh"
    environment = {
      ECR_URL = aws_ecr_repository.how-low-can-you-llm-ecr.repository_url
    }
  }
}
