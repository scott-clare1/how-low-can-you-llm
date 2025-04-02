provider "aws" {
  region = var.region
}

resource "aws_lb" "how-low-can-you-llm-alb" {
  name               = var.alb_name
  internal           = false
  load_balancer_type = "application"
  security_groups    = [var.security_group_id]
  subnets            = [var.subnet_id_a, var.subnet_id_b]
}

resource "aws_lb_target_group" "target_group" {
  name        = var.lb_target_group_name
  port        = 8080
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id
}

resource "aws_lb_listener" "listener" {
  load_balancer_arn = aws_lb.how-low-can-you-llm-alb.arn
  port              = 8080
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.target_group.arn
  }
}
