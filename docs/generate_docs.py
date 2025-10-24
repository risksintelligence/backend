"""
Simplified API Documentation Generator for RiskX Platform
Creates OpenAPI documentation without importing the full FastAPI app.
"""
import json
import yaml
from pathlib import Path
from datetime import datetime


def create_openapi_schema():
    """Create OpenAPI 3.0 schema manually."""
    
    schema = {
        "openapi": "3.0.0",
        "info": {
            "title": "RiskX Risk Intelligence Platform API",
            "version": "1.0.0",
            "description": """
# RiskX Risk Intelligence Platform API

A comprehensive risk assessment platform providing real-time analysis of economic, geopolitical, 
environmental, and supply chain risks through advanced machine learning models and data integration.

## Features

- **Real-time Risk Assessment**: Live risk scoring using ML models
- **Economic Data Integration**: FRED, BEA, BLS, Census API integration  
- **Geopolitical Analysis**: CISA cybersecurity and threat intelligence
- **Environmental Monitoring**: NOAA weather and climate risk assessment
- **Geological Hazards**: USGS earthquake and natural disaster tracking
- **Supply Chain Intelligence**: Transportation and logistics risk analysis
- **Network Analysis**: Risk propagation and vulnerability assessment
- **Intelligent Caching**: Three-tier caching for optimal performance

## Architecture

- **Backend**: FastAPI with async support
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Cache**: Redis + PostgreSQL + File system (3-tier)
- **ML Models**: Scikit-learn with real-time prediction serving
- **ETL Pipeline**: Apache Airflow for data processing
- **Deployment**: Render cloud platform

## Authentication

This is an **open research platform** - no authentication required.
All endpoints are publicly accessible for research and educational purposes.
            """,
            "contact": {
                "name": "RiskX Support",
                "email": "support@riskx.ai",
                "url": "https://docs.riskx.ai"
            },
            "license": {
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
            {
                "url": "https://api.riskx.ai",
                "description": "Production server"
            },
            {
                "url": "https://staging-api.riskx.ai",
                "description": "Staging server"
            },
            {
                "url": "http://localhost:8000",
                "description": "Local development server"
            }
        ],
        "tags": [
            {"name": "health", "description": "Health check and system status"},
            {"name": "risk", "description": "Core risk assessment and predictions"},
            {"name": "economic", "description": "Economic indicators and market data"},
            {"name": "external", "description": "External data source integrations"},
            {"name": "network", "description": "Risk network analysis and propagation"},
            {"name": "cache", "description": "Intelligent caching system management"},
            {"name": "database", "description": "Database operations and data management"},
            {"name": "websocket", "description": "Real-time data streaming"}
        ],
        "paths": {},
        "components": {
            "schemas": {
                "SuccessResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "example": "success"},
                        "data": {"type": "object"},
                        "source": {"type": "string", "example": "real_time"},
                        "timestamp": {"type": "string", "format": "date-time"}
                    }
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "example": "error"},
                        "error": {"type": "string"},
                        "timestamp": {"type": "string", "format": "date-time"}
                    }
                },
                "RiskOverview": {
                    "type": "object",
                    "properties": {
                        "overall_risk_score": {"type": "number", "example": 65.5},
                        "risk_level": {"type": "string", "example": "Medium-High"},
                        "confidence": {"type": "number", "example": 0.85},
                        "factors": {
                            "type": "object",
                            "properties": {
                                "economic": {"type": "number", "example": 70.0},
                                "geopolitical": {"type": "number", "example": 60.0},
                                "supply_chain": {"type": "number", "example": 65.0},
                                "environmental": {"type": "number", "example": 45.0}
                            }
                        },
                        "assessment_timestamp": {"type": "string", "format": "date-time"}
                    }
                },
                "EconomicIndicator": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "example": 27100000.0},
                        "units": {"type": "string", "example": "millions_usd"},
                        "date": {"type": "string", "example": "2024-Q2"},
                        "source": {"type": "string", "example": "fred"},
                        "change_pct": {"type": "number", "example": 2.1},
                        "trend": {"type": "string", "example": "growing"}
                    }
                },
                "SimulationRequest": {
                    "type": "object",
                    "required": ["initial_nodes", "shock_magnitude"],
                    "properties": {
                        "initial_nodes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "example": ["BANK_A"]
                        },
                        "shock_magnitude": {"type": "number", "example": 0.5},
                        "simulation_steps": {"type": "integer", "example": 20},
                        "containment_threshold": {"type": "number", "example": 0.1}
                    }
                }
            },
            "examples": {
                "RiskOverviewExample": {
                    "summary": "Risk Overview Response",
                    "value": {
                        "status": "success",
                        "data": {
                            "overall_risk_score": 65.5,
                            "risk_level": "Medium-High",
                            "confidence": 0.85,
                            "factors": {
                                "economic": 70.0,
                                "geopolitical": 60.0,
                                "supply_chain": 65.0,
                                "environmental": 45.0
                            },
                            "assessment_timestamp": "2024-01-01T12:00:00Z"
                        },
                        "source": "real_time",
                        "timestamp": "2024-01-01T12:00:00Z"
                    }
                }
            }
        }
    }
    
    # Add API endpoints
    add_api_endpoints(schema)
    
    return schema


def add_api_endpoints(schema):
    """Add API endpoint definitions to the schema."""
    
    paths = {
        "/": {
            "get": {
                "tags": ["health"],
                "summary": "Root endpoint",
                "description": "Get platform information and status",
                "responses": {
                    "200": {
                        "description": "Platform information",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SuccessResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/api/v1/health": {
            "get": {
                "tags": ["health"],
                "summary": "Health check",
                "description": "Comprehensive system health check",
                "responses": {
                    "200": {
                        "description": "System health status",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SuccessResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/api/v1/risk/overview": {
            "get": {
                "tags": ["risk"],
                "summary": "Get risk overview",
                "description": "Comprehensive risk assessment from all ML models",
                "responses": {
                    "200": {
                        "description": "Risk overview data",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SuccessResponse"},
                                "examples": {
                                    "example1": {"$ref": "#/components/examples/RiskOverviewExample"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/v1/risk/predictions/recession": {
            "get": {
                "tags": ["risk"],
                "summary": "Recession probability prediction",
                "description": "Get recession probability from ML model",
                "responses": {
                    "200": {
                        "description": "Recession prediction",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SuccessResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/api/v1/risk/predictions/supply-chain": {
            "get": {
                "tags": ["risk"],
                "summary": "Supply chain risk prediction",
                "description": "Get supply chain risk assessment from ML model",
                "responses": {
                    "200": {
                        "description": "Supply chain risk prediction",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SuccessResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/api/v1/risk/predictions/market-volatility": {
            "get": {
                "tags": ["risk"],
                "summary": "Market volatility prediction",
                "description": "Get market volatility prediction from ML model",
                "responses": {
                    "200": {
                        "description": "Market volatility prediction",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SuccessResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/api/v1/risk/predictions/geopolitical": {
            "get": {
                "tags": ["risk"],
                "summary": "Geopolitical risk prediction",
                "description": "Get geopolitical risk prediction from ML model",
                "responses": {
                    "200": {
                        "description": "Geopolitical risk prediction",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SuccessResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/api/v1/economic/indicators": {
            "get": {
                "tags": ["economic"],
                "summary": "Get economic indicators",
                "description": "Retrieve current economic indicators from multiple sources",
                "responses": {
                    "200": {
                        "description": "Economic indicators data",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SuccessResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/api/v1/network/simulation": {
            "post": {
                "tags": ["network"],
                "summary": "Run risk simulation",
                "description": "Execute shock propagation simulation on risk network",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/SimulationRequest"}
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Simulation results",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SuccessResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/api/v1/cache/metrics": {
            "get": {
                "tags": ["cache"],
                "summary": "Cache performance metrics",
                "description": "Get cache system performance metrics",
                "responses": {
                    "200": {
                        "description": "Cache metrics",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SuccessResponse"}
                            }
                        }
                    }
                }
            }
        }
    }
    
    schema["paths"] = paths


def generate_documentation():
    """Generate OpenAPI documentation files."""
    
    print("🚀 Generating RiskX API Documentation...")
    
    try:
        # Create docs directory
        docs_dir = Path(__file__).parent
        docs_dir.mkdir(exist_ok=True)
        
        # Generate OpenAPI schema
        openapi_schema = create_openapi_schema()
        
        # Write OpenAPI JSON
        json_file = docs_dir / "openapi.json"
        with open(json_file, "w") as f:
            json.dump(openapi_schema, f, indent=2)
        
        # Write OpenAPI YAML
        yaml_file = docs_dir / "openapi.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(openapi_schema, f, default_flow_style=False, sort_keys=False)
        
        # Generate endpoint summary
        generate_endpoint_summary(openapi_schema, docs_dir)
        
        print("✅ API documentation generated successfully!")
        print(f"📁 Files created:")
        print(f"   - OpenAPI JSON: {json_file}")
        print(f"   - OpenAPI YAML: {yaml_file}")
        print(f"   - README: {docs_dir / 'README.md'}")
        print(f"   - Endpoints: {docs_dir / 'endpoints.md'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating documentation: {e}")
        return False


def generate_endpoint_summary(schema, docs_dir):
    """Generate endpoint summary documentation."""
    
    endpoints_md = """# RiskX API Endpoints

## Complete Endpoint Reference

This document provides a comprehensive list of all available API endpoints.

"""
    
    for path, methods in schema["paths"].items():
        endpoints_md += f"### `{path}`\n\n"
        
        for method, details in methods.items():
            method_upper = method.upper()
            summary = details.get("summary", "")
            description = details.get("description", "")
            tags = ", ".join(details.get("tags", []))
            
            endpoints_md += f"**{method_upper}** - {summary}\n\n"
            
            if description:
                endpoints_md += f"{description}\n\n"
            
            if tags:
                endpoints_md += f"*Tags: {tags}*\n\n"
            
            # Add request body info if present
            if "requestBody" in details:
                endpoints_md += "**Request Body Required**\n\n"
            
            # Add response codes
            responses = details.get("responses", {})
            if responses:
                endpoints_md += "**Responses:**\n"
                for code, response_info in responses.items():
                    desc = response_info.get("description", "")
                    endpoints_md += f"- `{code}`: {desc}\n"
                endpoints_md += "\n"
            
            endpoints_md += "---\n\n"
    
    # Write endpoints documentation
    with open(docs_dir / "endpoints.md", "w") as f:
        f.write(endpoints_md)


if __name__ == "__main__":
    generate_documentation()