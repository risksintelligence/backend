# RiskX Terraform Variables
# Risk Intelligence Observatory Infrastructure Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "riskx"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

# Networking Variables
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid CIDR block."
  }
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  validation {
    condition     = length(var.public_subnet_cidrs) >= 2
    error_message = "At least 2 public subnets are required for high availability."
  }
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.20.0/24", "10.0.30.0/24"]
  validation {
    condition     = length(var.private_subnet_cidrs) >= 2
    error_message = "At least 2 private subnets are required for high availability."
  }
}

# Database Variables
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_allocated_storage" {
  description = "Initial allocated storage for RDS (GB)"
  type        = number
  default     = 100
  validation {
    condition     = var.db_allocated_storage >= 20
    error_message = "Allocated storage must be at least 20 GB."
  }
}

variable "db_max_allocated_storage" {
  description = "Maximum allocated storage for RDS (GB)"
  type        = number
  default     = 1000
  validation {
    condition     = var.db_max_allocated_storage >= var.db_allocated_storage
    error_message = "Max allocated storage must be greater than or equal to initial allocated storage."
  }
}

variable "db_backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
  validation {
    condition     = var.db_backup_retention_period >= 1 && var.db_backup_retention_period <= 35
    error_message = "Backup retention period must be between 1 and 35 days."
  }
}

# Cache Variables
variable "cache_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.medium"
}

variable "cache_num_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 2
  validation {
    condition     = var.cache_num_nodes >= 1 && var.cache_num_nodes <= 6
    error_message = "Number of cache nodes must be between 1 and 6."
  }
}

# API Keys (Free Government APIs)
variable "fred_api_key" {
  description = "Federal Reserve Economic Data API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "census_api_key" {
  description = "U.S. Census Bureau API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "bea_api_key" {
  description = "Bureau of Economic Analysis API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "bls_api_key" {
  description = "Bureau of Labor Statistics API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "noaa_api_key" {
  description = "NOAA Climate Data API key"
  type        = string
  sensitive   = true
  default     = ""
}

# EKS Variables
variable "eks_cluster_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "eks_node_group_instance_types" {
  description = "Instance types for EKS node group"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "eks_node_group_desired_size" {
  description = "Desired number of nodes in EKS node group"
  type        = number
  default     = 3
}

variable "eks_node_group_max_size" {
  description = "Maximum number of nodes in EKS node group"
  type        = number
  default     = 10
}

variable "eks_node_group_min_size" {
  description = "Minimum number of nodes in EKS node group"
  type        = number
  default     = 1
}

# Application Variables
variable "app_replicas" {
  description = "Number of application replicas"
  type        = number
  default     = 3
  validation {
    condition     = var.app_replicas >= 1
    error_message = "Number of application replicas must be at least 1."
  }
}

variable "app_cpu_request" {
  description = "CPU request for application containers"
  type        = string
  default     = "500m"
}

variable "app_memory_request" {
  description = "Memory request for application containers"
  type        = string
  default     = "512Mi"
}

variable "app_cpu_limit" {
  description = "CPU limit for application containers"
  type        = string
  default     = "1000m"
}

variable "app_memory_limit" {
  description = "Memory limit for application containers"
  type        = string
  default     = "1Gi"
}

# Domain and SSL Variables
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "riskx.observatory"
}

variable "create_route53_zone" {
  description = "Create Route53 hosted zone"
  type        = bool
  default     = false
}

variable "ssl_certificate_arn" {
  description = "ARN of SSL certificate in ACM"
  type        = string
  default     = ""
}

# Monitoring Variables
variable "enable_cloudwatch_insights" {
  description = "Enable CloudWatch Container Insights"
  type        = bool
  default     = true
}

variable "enable_prometheus" {
  description = "Enable Prometheus monitoring"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Enable Grafana dashboards"
  type        = bool
  default     = true
}

# Backup Variables
variable "enable_s3_backup" {
  description = "Enable S3 backup for data"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
  validation {
    condition     = var.backup_retention_days >= 1
    error_message = "Backup retention days must be at least 1."
  }
}

# Security Variables
variable "enable_waf" {
  description = "Enable AWS WAF"
  type        = bool
  default     = true
}

variable "enable_shield" {
  description = "Enable AWS Shield Advanced"
  type        = bool
  default     = false
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the application"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Cost Optimization Variables
variable "enable_spot_instances" {
  description = "Use spot instances for EKS nodes"
  type        = bool
  default     = false
}

variable "enable_autoscaling" {
  description = "Enable cluster autoscaling"
  type        = bool
  default     = true
}

# Compliance Variables
variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest for all resources"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Enable encryption in transit"
  type        = bool
  default     = true
}

variable "enable_audit_logging" {
  description = "Enable audit logging"
  type        = bool
  default     = true
}

# Professional Standards Variables
variable "professional_color_scheme" {
  description = "Professional color scheme configuration"
  type = object({
    primary_navy   = string
    charcoal_gray  = string
    pure_white     = string
  })
  default = {
    primary_navy  = "#1e3a8a"
    charcoal_gray = "#374151"
    pure_white    = "#ffffff"
  }
}

# Risk Intelligence Specific Variables
variable "risk_calculation_interval" {
  description = "Risk calculation interval in seconds"
  type        = number
  default     = 300
  validation {
    condition     = var.risk_calculation_interval >= 60
    error_message = "Risk calculation interval must be at least 60 seconds."
  }
}

variable "alert_thresholds" {
  description = "Risk alert thresholds"
  type = object({
    critical = number
    high     = number
    moderate = number
  })
  default = {
    critical = 80
    high     = 60
    moderate = 40
  }
  validation {
    condition = (
      var.alert_thresholds.critical > var.alert_thresholds.high &&
      var.alert_thresholds.high > var.alert_thresholds.moderate &&
      var.alert_thresholds.moderate > 0
    )
    error_message = "Alert thresholds must be in descending order and greater than 0."
  }
}

variable "data_retention_days" {
  description = "Data retention period in days"
  type        = number
  default     = 365
  validation {
    condition     = var.data_retention_days >= 30
    error_message = "Data retention must be at least 30 days."
  }
}

variable "cache_ttl_seconds" {
  description = "Cache TTL in seconds"
  type        = number
  default     = 3600
  validation {
    condition     = var.cache_ttl_seconds >= 60
    error_message = "Cache TTL must be at least 60 seconds."
  }
}

# Feature Flags
variable "feature_flags" {
  description = "Feature flags for enabling/disabling functionality"
  type = object({
    enable_ml_models         = bool
    enable_explainable_ai    = bool
    enable_bias_detection    = bool
    enable_real_time_alerts  = bool
    enable_data_export       = bool
    enable_api_rate_limiting = bool
  })
  default = {
    enable_ml_models         = true
    enable_explainable_ai    = true
    enable_bias_detection    = true
    enable_real_time_alerts  = true
    enable_data_export       = true
    enable_api_rate_limiting = true
  }
}

# Tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}