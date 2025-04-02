module "networking" {
  source = "./networking"

  region = var.region
}

module "ecr" {
  source = "./ecr"

  region   = var.region
  ecr_name = var.ecr_name
}

module "ec2" {
  source = "./ec2"

  region       = var.region
  cluster_name = var.ecs_cluster_name
  subnet_id_a  = module.networking.subnet_id_a
  subnet_id_b  = module.networking.subnet_id_b
}

module "alb" {
  source = "./alb"

  region            = var.region
  security_group_id = module.networking.security_group_id
  subnet_id_a       = module.networking.subnet_id_a
  subnet_id_b       = module.networking.subnet_id_b
  vpc_id            = module.networking.vpc_id
}

module "ecs" {
  source = "./ecs"

  region                         = var.region
  aws_account_id                 = var.aws_account_id
  ecr_name                       = module.ecr.ecr_name
  image_name                     = var.image_name
  security_group_id              = module.networking.security_group_id
  subnet_id_a                    = module.networking.subnet_id_a
  subnet_id_b                    = module.networking.subnet_id_b
  autoscaler_arn                 = module.ec2.autoscaler_arn
  ecs_cluster_name               = var.ecs_cluster_name
  load_balancer_target_group_arn = module.alb.alb_target_group_arn
}
