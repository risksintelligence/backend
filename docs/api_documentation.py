"""
API Documentation Generator for RiskX Platform
Generates comprehensive OpenAPI/Swagger documentation.
"""
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import json
import yaml
from pathlib import Path


def generate_custom_openapi(app: FastAPI) -> dict:
    """Generate custom OpenAPI schema with enhanced documentation."""
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="RiskX Risk Intelligence Platform API",
        version="1.0.0",
        description="""
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

## Rate Limiting

- Standard rate limiting applies to prevent abuse
- Contact support for higher rate limits if needed

## Data Sources

- **FRED**: Federal Reserve Economic Data
- **BEA**: Bureau of Economic Analysis
- **BLS**: Bureau of Labor Statistics  
- **Census**: U.S. Census Bureau
- **CISA**: Cybersecurity & Infrastructure Security Agency
- **NOAA**: National Oceanic and Atmospheric Administration
- **USGS**: U.S. Geological Survey

## Support

For questions, issues, or contributions:
- GitHub: https://github.com/riskx-platform
- Documentation: https://docs.riskx.ai
- Support: support@riskx.ai
        """,
        routes=app.routes,
        tags=[
            {
                "name": "Health",
                "description": "Health check and system status endpoints"
            },
            {
                "name": "Risk Assessment",
                "description": "Core risk assessment and prediction endpoints"
            },
            {
                "name": "Machine Learning",
                "description": "ML model predictions and management"
            },
            {
                "name": "Economic Data",
                "description": "Economic indicators and market data"
            },
            {
                "name": "External APIs",
                "description": "External data source integrations"
            },
            {
                "name": "Network Analysis",
                "description": "Risk network analysis and propagation"
            },
            {
                "name": "Cache Management",
                "description": "Intelligent caching system management"
            },
            {
                "name": "Database",
                "description": "Database operations and data management"
            },
            {
                "name": "WebSocket",
                "description": "Real-time data streaming via WebSocket"
            }
        ]
    )
    
    # Add custom fields
    openapi_schema["info"]["contact"] = {
        "name": "RiskX Support",
        "email": "support@riskx.ai",
        "url": "https://docs.riskx.ai"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    openapi_schema["servers"] = [
        {
            "url": "https://api.riskx.ai",
            "description": "Production server"
        },
    ]
    
    # Add response examples
    add_response_examples(openapi_schema)
    
    # Add security schemes (even though not used, for completeness)
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key authentication (not required for this open platform)"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def add_response_examples(schema: dict):
    """Add response examples to the OpenAPI schema."""
    
    # Risk overview response example
    risk_overview_example = {
        "status": "success",
        "data": {
            "overall_risk_score": 65.5,
            "risk_level": "Medium-High",
            "confidence": 0.85,
            "factors": {
                "economic": {
                    "score": 70.0,
                    "trend": "rising",
                    "key_indicators": ["GDP growth", "unemployment rate", "inflation"]
                },
                "geopolitical": {
                    "score": 60.0,
                    "trend": "stable", 
                    "key_factors": ["trade tensions", "sanctions", "conflicts"]
                },
                "supply_chain": {
                    "score": 65.0,
                    "trend": "declining",
                    "key_risks": ["shipping costs", "port congestion", "labor shortages"]
                },
                "environmental": {
                    "score": 45.0,
                    "trend": "rising",
                    "key_hazards": ["severe weather", "climate change", "natural disasters"]
                }
            },
            "recommendations": [
                "Monitor supply chain disruptions closely",
                "Hedge against currency fluctuations",
                "Diversify geographic exposure"
            ],
            "assessment_timestamp": "2024-01-01T12:00:00Z",
            "models_used": 4,
            "data_sources": ["fred", "bea", "bls", "cisa", "noaa", "usgs"]
        },
        "source": "real_time",
        "timestamp": "2024-01-01T12:00:00Z"
    }
    
    # Economic indicators response example
    economic_indicators_example = {
        "status": "success",
        "data": {
            "indicators": {
                "GDP": {
                    "value": 27100000.0,
                    "units": "millions_usd",
                    "date": "2024-Q2",
                    "source": "fred",
                    "change_pct": 2.1,
                    "trend": "growing"
                },
                "UNRATE": {
                    "value": 3.4,
                    "units": "percent",
                    "date": "2024-07-01",
                    "source": "bls",
                    "change_pct": -0.1,
                    "trend": "declining"
                },
                "DGS10": {
                    "value": 4.2,
                    "units": "percent",
                    "date": "2024-07-30",
                    "source": "fred",
                    "change_pct": 0.3,
                    "trend": "rising"
                }
            },
            "summary": {
                "total_indicators": 15,
                "last_updated": "2024-01-01T12:00:00Z",
                "data_freshness": "excellent"
            }
        },
        "count": 15,
        "timestamp": "2024-01-01T12:00:00Z"
    }
    
    # Network analysis response example
    network_analysis_example = {
        "status": "success",
        "data": {
            "network_metrics": {
                "total_nodes": 250,
                "total_edges": 847,
                "average_degree": 6.78,
                "clustering_coefficient": 0.42,
                "network_density": 0.027
            },
            "critical_components": [
                {
                    "node_id": "BANK_JPM",
                    "name": "JPMorgan Chase",
                    "centrality_score": 0.89,
                    "risk_level": 65.0,
                    "connections": 45,
                    "systemic_importance": "critical"
                },
                {
                    "node_id": "TECH_AAPL", 
                    "name": "Apple Inc",
                    "centrality_score": 0.76,
                    "risk_level": 42.0,
                    "connections": 38,
                    "systemic_importance": "high"
                }
            ],
            "vulnerability_assessment": {
                "overall_resilience": 72.0,
                "single_point_failures": 3,
                "cascade_risk": "medium",
                "recovery_time_estimate": "2-4 weeks"
            }
        },
        "timestamp": "2024-01-01T12:00:00Z"
    }
    
    # Store examples in schema
    if "components" not in schema:
        schema["components"] = {}
    if "examples" not in schema["components"]:
        schema["components"]["examples"] = {}
    
    schema["components"]["examples"].update({
        "RiskOverviewResponse": {
            "summary": "Risk Overview Response",
            "value": risk_overview_example
        },
        "EconomicIndicatorsResponse": {
            "summary": "Economic Indicators Response", 
            "value": economic_indicators_example
        },
        "NetworkAnalysisResponse": {
            "summary": "Network Analysis Response",
            "value": network_analysis_example
        }
    })


def generate_api_docs():
    """Generate API documentation files."""
    
    # Import the FastAPI app
    import sys
    from pathlib import Path
    
    # Add the backend directory to Python path
    backend_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(backend_dir))
    
    try:
        from src.api.main import app
        
        # Generate OpenAPI schema
        openapi_schema = generate_custom_openapi(app)
        
        # Create docs directory
        docs_dir = Path(__file__).parent
        docs_dir.mkdir(exist_ok=True)
        
        # Write OpenAPI JSON
        with open(docs_dir / "openapi.json", "w") as f:
            json.dump(openapi_schema, f, indent=2)
        
        # Write OpenAPI YAML
        with open(docs_dir / "openapi.yaml", "w") as f:
            yaml.dump(openapi_schema, f, default_flow_style=False, sort_keys=False)
        
        print("API documentation generated successfully!")
        print(f"OpenAPI JSON: {docs_dir / 'openapi.json'}")
        print(f"OpenAPI YAML: {docs_dir / 'openapi.yaml'}")
        
        return True
        
    except Exception as e:
        print(f"Error generating API documentation: {e}")
        return False


if __name__ == "__main__":
    generate_api_docs()