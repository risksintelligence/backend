# RiskX Risk Intelligence Observatory Infrastructure
# Terraform configuration for production deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  backend "s3" {
    bucket         = "riskx-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "riskx-terraform-locks"
  }
}

# Configure the AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "RiskX Observatory"
      Environment = var.environment
      Owner       = "Risk Intelligence Team"
      Purpose     = "Financial Risk Analysis Platform"
      Compliance  = "SOC2-TypeII,GDPR,CCPA"
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Random password generation
resource "random_password" "database_password" {
  length  = 32
  special = true
}

resource "random_password" "redis_password" {
  length  = 32
  special = false
}

resource "random_password" "secret_key" {
  length  = 64
  special = true
}

# VPC Configuration
resource "aws_vpc" "riskx_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "riskx_igw" {
  vpc_id = aws_vpc.riskx_vpc.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

# Public Subnets
resource "aws_subnet" "public_subnets" {
  count                   = length(var.public_subnet_cidrs)
  vpc_id                  = aws_vpc.riskx_vpc.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-public-subnet-${count.index + 1}"
    Type = "public"
  }
}

# Private Subnets
resource "aws_subnet" "private_subnets" {
  count             = length(var.private_subnet_cidrs)
  vpc_id            = aws_vpc.riskx_vpc.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "${var.project_name}-private-subnet-${count.index + 1}"
    Type = "private"
  }
}

# NAT Gateways
resource "aws_eip" "nat_eips" {
  count  = length(aws_subnet.public_subnets)
  domain = "vpc"

  tags = {
    Name = "${var.project_name}-nat-eip-${count.index + 1}"
  }
}

resource "aws_nat_gateway" "nat_gateways" {
  count         = length(aws_subnet.public_subnets)
  allocation_id = aws_eip.nat_eips[count.index].id
  subnet_id     = aws_subnet.public_subnets[count.index].id

  tags = {
    Name = "${var.project_name}-nat-gateway-${count.index + 1}"
  }

  depends_on = [aws_internet_gateway.riskx_igw]
}

# Route Tables
resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.riskx_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.riskx_igw.id
  }

  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

resource "aws_route_table" "private_rt" {
  count  = length(aws_nat_gateway.nat_gateways)
  vpc_id = aws_vpc.riskx_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_gateways[count.index].id
  }

  tags = {
    Name = "${var.project_name}-private-rt-${count.index + 1}"
  }
}

# Route Table Associations
resource "aws_route_table_association" "public_rta" {
  count          = length(aws_subnet.public_subnets)
  subnet_id      = aws_subnet.public_subnets[count.index].id
  route_table_id = aws_route_table.public_rt.id
}

resource "aws_route_table_association" "private_rta" {
  count          = length(aws_subnet.private_subnets)
  subnet_id      = aws_subnet.private_subnets[count.index].id
  route_table_id = aws_route_table.private_rt[count.index].id
}

# Security Groups
resource "aws_security_group" "alb_sg" {
  name_prefix = "${var.project_name}-alb-"
  vpc_id      = aws_vpc.riskx_vpc.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-alb-sg"
  }
}

resource "aws_security_group" "eks_sg" {
  name_prefix = "${var.project_name}-eks-"
  vpc_id      = aws_vpc.riskx_vpc.id

  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-eks-sg"
  }
}

resource "aws_security_group" "rds_sg" {
  name_prefix = "${var.project_name}-rds-"
  vpc_id      = aws_vpc.riskx_vpc.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_sg.id]
  }

  tags = {
    Name = "${var.project_name}-rds-sg"
  }
}

resource "aws_security_group" "elasticache_sg" {
  name_prefix = "${var.project_name}-elasticache-"
  vpc_id      = aws_vpc.riskx_vpc.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_sg.id]
  }

  tags = {
    Name = "${var.project_name}-elasticache-sg"
  }
}

# RDS Subnet Group
resource "aws_db_subnet_group" "riskx_db_subnet_group" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = aws_subnet.private_subnets[*].id

  tags = {
    Name = "${var.project_name}-db-subnet-group"
  }
}

# ElastiCache Subnet Group
resource "aws_elasticache_subnet_group" "riskx_cache_subnet_group" {
  name       = "${var.project_name}-cache-subnet-group"
  subnet_ids = aws_subnet.private_subnets[*].id

  tags = {
    Name = "${var.project_name}-cache-subnet-group"
  }
}

# RDS PostgreSQL Instance
resource "aws_db_instance" "riskx_db" {
  identifier     = "${var.project_name}-db"
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.db_instance_class
  
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "riskx"
  username = "riskx_admin"
  password = random_password.database_password.result
  
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.riskx_db_subnet_group.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.project_name}-db-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
  
  deletion_protection = var.environment == "production" ? true : false
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  
  tags = {
    Name = "${var.project_name}-database"
  }
}

# ElastiCache Redis Cluster
resource "aws_elasticache_replication_group" "riskx_cache" {
  replication_group_id       = "${var.project_name}-cache"
  description                = "Redis cluster for RiskX caching"
  
  node_type                  = var.cache_node_type
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = var.cache_num_nodes
  automatic_failover_enabled = var.cache_num_nodes > 1
  multi_az_enabled          = var.cache_num_nodes > 1
  
  subnet_group_name = aws_elasticache_subnet_group.riskx_cache_subnet_group.name
  security_group_ids = [aws_security_group.elasticache_sg.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_password.result
  
  snapshot_retention_limit = 5
  snapshot_window         = "03:00-05:00"
  
  tags = {
    Name = "${var.project_name}-cache"
  }
}

# S3 Bucket for Data Storage
resource "aws_s3_bucket" "riskx_data_bucket" {
  bucket = "${var.project_name}-data-${random_string.bucket_suffix.result}"

  tags = {
    Name = "${var.project_name}-data-bucket"
  }
}

resource "aws_s3_bucket_versioning" "riskx_data_bucket_versioning" {
  bucket = aws_s3_bucket.riskx_data_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "riskx_data_bucket_encryption" {
  bucket = aws_s3_bucket.riskx_data_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "riskx_data_bucket_pab" {
  bucket = aws_s3_bucket.riskx_data_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# Application Load Balancer
resource "aws_lb" "riskx_alb" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = aws_subnet.public_subnets[*].id

  enable_deletion_protection = var.environment == "production" ? true : false

  access_logs {
    bucket  = aws_s3_bucket.riskx_logs_bucket.bucket
    prefix  = "alb-logs"
    enabled = true
  }

  tags = {
    Name = "${var.project_name}-alb"
  }
}

# S3 Bucket for Logs
resource "aws_s3_bucket" "riskx_logs_bucket" {
  bucket = "${var.project_name}-logs-${random_string.bucket_suffix.result}"

  tags = {
    Name = "${var.project_name}-logs-bucket"
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "riskx_app_logs" {
  name              = "/aws/riskx/application"
  retention_in_days = 30

  tags = {
    Name = "${var.project_name}-app-logs"
  }
}

resource "aws_cloudwatch_log_group" "riskx_api_logs" {
  name              = "/aws/riskx/api"
  retention_in_days = 30

  tags = {
    Name = "${var.project_name}-api-logs"
  }
}

# Secrets Manager
resource "aws_secretsmanager_secret" "riskx_secrets" {
  name                    = "${var.project_name}-secrets"
  description             = "RiskX application secrets"
  recovery_window_in_days = 7

  tags = {
    Name = "${var.project_name}-secrets"
  }
}

resource "aws_secretsmanager_secret_version" "riskx_secrets_version" {
  secret_id = aws_secretsmanager_secret.riskx_secrets.id
  secret_string = jsonencode({
    database_password = random_password.database_password.result
    redis_password    = random_password.redis_password.result
    secret_key        = random_password.secret_key.result
    fred_api_key      = var.fred_api_key
    census_api_key    = var.census_api_key
    bea_api_key       = var.bea_api_key
    bls_api_key       = var.bls_api_key
  })
}

# IAM Roles and Policies
resource "aws_iam_role" "riskx_app_role" {
  name = "${var.project_name}-app-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.project_name}-app-role"
  }
}

resource "aws_iam_policy" "riskx_app_policy" {
  name        = "${var.project_name}-app-policy"
  description = "Policy for RiskX application"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.riskx_data_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.riskx_secrets.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = [
          aws_cloudwatch_log_group.riskx_app_logs.arn,
          aws_cloudwatch_log_group.riskx_api_logs.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "riskx_app_policy_attachment" {
  role       = aws_iam_role.riskx_app_role.name
  policy_arn = aws_iam_policy.riskx_app_policy.arn
}

# Output values
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.riskx_vpc.id
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.riskx_db.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.riskx_cache.primary_endpoint_address
  sensitive   = true
}

output "alb_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.riskx_alb.dns_name
}

output "s3_data_bucket" {
  description = "Name of the S3 data bucket"
  value       = aws_s3_bucket.riskx_data_bucket.bucket
}

output "secrets_manager_arn" {
  description = "ARN of the Secrets Manager secret"
  value       = aws_secretsmanager_secret.riskx_secrets.arn
}