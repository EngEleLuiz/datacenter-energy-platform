variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "dc-energy-platform"
}

variable "environment" {
  type    = string
  default = "dev"
}

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}
```

---

## `.gitignore`
```
__pycache__/
*.py[cod]
.env
.venv/
*.parquet
*.csv
*.pkl
mlruns/
.ipynb_checkpoints/
infra/.terraform/
infra/terraform.tfstate*
*.log
.DS_Store