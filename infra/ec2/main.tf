provider "aws" {
  region = var.region
}

resource "aws_launch_template" "ecs_lt" {
  image_id      = var.ami
  instance_type = var.instance_type

  iam_instance_profile {
    name = "ecsInstanceRole"
  }

  user_data = filebase64("${path.module}/ecs.sh")

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = 30
      volume_type = "gp2"
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "ecs-instance"
    }
  }
}

resource "aws_autoscaling_group" "ecs_asg" {
  vpc_zone_identifier = [var.subnet_id_a, var.subnet_id_b]
  desired_capacity    = 2
  max_size            = 3
  min_size            = 1

  launch_template {
    id      = aws_launch_template.ecs_lt.id
    version = "$Latest"
  }

  tag {
    key                 = "AmazonECSManaged"
    value               = true
    propagate_at_launch = true
  }
}
