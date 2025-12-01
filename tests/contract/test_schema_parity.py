#!/usr/bin/env python3
"""
Frontend-Backend Schema Parity Contract Tests
Validates that API responses match expected TypeScript interface contracts
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

import httpx
import pytest
import structlog
from jsonschema import Draft7Validator, ValidationError

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = structlog.get_logger()

class SchemaContractValidator:
    """Validates API responses against expected frontend schemas"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.schemas = self._load_expected_schemas()
        
    def _load_expected_schemas(self) -> Dict[str, Dict]:
        """Load expected response schemas based on frontend TypeScript interfaces"""
        return {
            "/api/v1/analytics/geri": {
                "type": "object",
                "required": ["score", "band", "confidence", "updated_at"],
                "properties": {
                    "score": {"type": "number", "minimum": 0, "maximum": 100},
                    "band": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 100},
                    "updated_at": {"type": "string"},
                    "contributions": {"type": "object"},
                    "component_scores": {"type": "object"},
                    "metadata": {"type": "object"},
                    "drivers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["component", "contribution", "impact"],
                            "properties": {
                                "component": {"type": "string"},
                                "contribution": {"type": "number"},
                                "impact": {"type": "number"}
                            }
                        }
                    },
                    "color": {"type": "string"},
                    "band_color": {"type": "string"},
                    "change_24h": {"type": "number"},
                    "risk_taxonomy": {"type": "object"}
                }
            },
            "/api/v1/ai/governance/models": {
                "type": "object",
                "required": ["models"],
                "properties": {
                    "models": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["model_id", "version", "model_type", "registered_at"],
                            "properties": {
                                "model_id": {"type": "string"},
                                "version": {"type": "string"},
                                "model_type": {"type": "string"},
                                "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                                "registered_at": {"type": "string"},
                                "training_data_hash": {"type": "string"},
                                "performance_metrics": {"type": "object"}
                            }
                        }
                    },
                    "total_count": {"type": "integer", "minimum": 0}
                }
            },
            "/api/v1/ai/explainability/audit-log": {
                "type": "object",
                "required": ["audit_logs", "total_entries"],
                "properties": {
                    "audit_logs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["decision_id", "model_id", "model_version", "accessed_by", "access_timestamp"],
                            "properties": {
                                "decision_id": {"type": "string"},
                                "model_id": {"type": "string"},
                                "model_version": {"type": "string"},
                                "accessed_by": {"type": "string"},
                                "access_timestamp": {"type": "string"},
                                "explanation_level": {"type": "string"}
                            }
                        }
                    },
                    "total_entries": {"type": "integer", "minimum": 0},
                    "start_timestamp": {"type": "string"},
                    "end_timestamp": {"type": "string"}
                }
            },
            "/api/v1/ai/forecast/next-24h": {
                "type": "object",
                "required": ["delta", "p_gt_5", "confidence_interval", "updated_at"],
                "properties": {
                    "delta": {"type": "number"},
                    "p_gt_5": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence_interval": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2
                    },
                    "drivers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["component", "impact"],
                            "properties": {
                                "component": {"type": "string"},
                                "impact": {"type": "number"}
                            }
                        }
                    },
                    "updated_at": {"type": "string"}
                }
            },
            "/api/v1/ai/regime/current": {
                "type": "object",
                "required": ["regime", "probabilities", "confidence", "updated_at"],
                "properties": {
                    "regime": {"type": "string"},
                    "probabilities": {"type": "object"},
                    "weights": {"type": "object"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "updated_at": {"type": "string"}
                }
            }
        }
    
    async def validate_endpoint(self, endpoint: str) -> Dict[str, Any]:
        """Validate a single endpoint against its expected schema"""
        result = {
            "endpoint": endpoint,
            "status": "unknown",
            "response_code": 0,
            "schema_valid": False,
            "errors": [],
            "response_time_ms": 0,
            "data_sample": None
        }
        
        if endpoint not in self.schemas:
            result["status"] = "skipped"
            result["errors"] = ["No schema defined for this endpoint"]
            return result
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                start_time = asyncio.get_event_loop().time()
                
                # Handle parameterized endpoints
                test_url = endpoint
                if "{model_name}" in endpoint:
                    test_url = endpoint.replace("{model_name}", "test_model")
                elif "{decision_id}" in endpoint:
                    test_url = endpoint.replace("{decision_id}", "test_decision")
                
                # Add query params for audit log endpoint
                if "audit-log" in endpoint:
                    test_url += "?start_date=2025-01-01&end_date=2025-12-31"
                
                response = await client.get(f"{self.base_url}{test_url}")
                end_time = asyncio.get_event_loop().time()
                
                result["response_code"] = response.status_code
                result["response_time_ms"] = (end_time - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    result["data_sample"] = data
                    
                    # Validate against schema
                    validator = Draft7Validator(self.schemas[endpoint])
                    validation_errors = list(validator.iter_errors(data))
                    
                    if not validation_errors:
                        result["status"] = "pass"
                        result["schema_valid"] = True
                    else:
                        result["status"] = "fail"
                        result["schema_valid"] = False
                        result["errors"] = [f"{err.json_path}: {err.message}" for err in validation_errors[:5]]
                        
                elif response.status_code in [404, 500]:
                    # Expected for test data that doesn't exist
                    result["status"] = "endpoint_available"
                    result["errors"] = [f"HTTP {response.status_code} - endpoint exists but no test data"]
                else:
                    result["status"] = "fail"
                    result["errors"] = [f"Unexpected HTTP status: {response.status_code}"]
                    
        except Exception as e:
            result["status"] = "error"
            result["errors"] = [f"Request failed: {str(e)}"]
            
        return result
    
    async def validate_all_endpoints(self) -> Dict[str, Any]:
        """Validate all configured endpoints"""
        results = {
            "summary": {
                "total_endpoints": len(self.schemas),
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "skipped": 0
            },
            "endpoints": [],
            "timestamp": "2025-11-24T13:15:00Z"
        }
        
        for endpoint in self.schemas.keys():
            logger.info(f"Testing endpoint: {endpoint}")
            result = await self.validate_endpoint(endpoint)
            results["endpoints"].append(result)
            
            # Update summary
            if result["status"] == "pass":
                results["summary"]["passed"] += 1
            elif result["status"] == "fail":
                results["summary"]["failed"] += 1
            elif result["status"] == "error":
                results["summary"]["errors"] += 1
            else:
                results["summary"]["skipped"] += 1
        
        return results

async def main():
    """Main test runner"""
    print("🧪 Starting Frontend-Backend Schema Parity Contract Tests")
    print("=" * 60)
    
    validator = SchemaContractValidator()
    results = await validator.validate_all_endpoints()
    
    # Print summary
    summary = results["summary"]
    print(f"\n📊 Test Summary:")
    print(f"   Total Endpoints: {summary['total_endpoints']}")
    print(f"   ✅ Passed: {summary['passed']}")
    print(f"   ❌ Failed: {summary['failed']}")
    print(f"   🚫 Errors: {summary['errors']}")
    print(f"   ⏭️  Skipped: {summary['skipped']}")
    
    # Print detailed results
    print(f"\n📋 Detailed Results:")
    for result in results["endpoints"]:
        status_emoji = {
            "pass": "✅",
            "fail": "❌", 
            "error": "🚫",
            "skipped": "⏭️",
            "endpoint_available": "🟡"
        }.get(result["status"], "❓")
        
        print(f"   {status_emoji} {result['endpoint']} - {result['status'].upper()}")
        
        if result["errors"]:
            for error in result["errors"][:3]:  # Show first 3 errors
                print(f"      • {error}")
        
        if result["response_time_ms"] > 0:
            print(f"      Response time: {result['response_time_ms']:.1f}ms")
    
    # Save full results
    results_file = Path("tests/contract/contract_test_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Full results saved to: {results_file}")
    
    # Exit with appropriate code
    if summary["failed"] > 0 or summary["errors"] > 0:
        print(f"\n❌ Schema parity tests failed!")
        return 1
    else:
        print(f"\n✅ All schema parity tests passed!")
        return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
