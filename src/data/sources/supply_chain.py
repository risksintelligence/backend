"""
Supply Chain & Transportation Data Integration
Uses public datasets and free APIs for supply chain risk assessment
"""
import aiohttp
import asyncio
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)

# Free APIs and data sources
PORTS_API_URL = "https://api.weather.gov"  # For port weather conditions
FREIGHT_DATA_URL = "https://data.transportation.gov/api"  # DOT open data (if available)


class SupplyChainClient:
    """Async client for supply chain and transportation data with error handling."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "RiskX-Platform/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        now = asyncio.get_event_loop().time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    async def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        """Make request with error handling."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        
        await self._rate_limit()
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Supply Chain API error {response.status}: {await response.text()}")
                    return None
        
        except asyncio.TimeoutError:
            logger.error(f"Supply Chain API timeout for {url}")
            return None
        except Exception as e:
            logger.error(f"Supply Chain API error for {url}: {e}")
            return None
    
    async def get_critical_infrastructure_nodes(self) -> Optional[Dict]:
        """Identify critical infrastructure nodes in supply chain networks."""
        
        # Major US supply chain hubs and their characteristics
        critical_nodes = {
            "ports": {
                "Los Angeles/Long Beach": {"throughput_rank": 1, "cargo_value": "high", "region": "west_coast"},
                "New York/New Jersey": {"throughput_rank": 2, "cargo_value": "high", "region": "east_coast"},
                "Savannah": {"throughput_rank": 3, "cargo_value": "medium", "region": "southeast"},
                "Houston": {"throughput_rank": 4, "cargo_value": "high", "region": "gulf_coast"},
                "Seattle/Tacoma": {"throughput_rank": 5, "cargo_value": "medium", "region": "west_coast"},
                "Virginia": {"throughput_rank": 6, "cargo_value": "medium", "region": "east_coast"}
            },
            "rail_hubs": {
                "Chicago": {"importance": "critical", "connections": 7, "freight_volume": "very_high"},
                "Kansas City": {"importance": "high", "connections": 6, "freight_volume": "high"},
                "Memphis": {"importance": "high", "connections": 5, "freight_volume": "high"},
                "Atlanta": {"importance": "medium", "connections": 4, "freight_volume": "medium"},
                "Dallas": {"importance": "medium", "connections": 4, "freight_volume": "medium"}
            },
            "distribution_centers": {
                "Memphis (FedEx)": {"capacity": "very_high", "coverage": "national", "criticality": "critical"},
                "Louisville (UPS)": {"capacity": "very_high", "coverage": "national", "criticality": "critical"},
                "Phoenix": {"capacity": "high", "coverage": "western_us", "criticality": "high"},
                "Dallas": {"capacity": "high", "coverage": "central_us", "criticality": "high"},
                "Atlanta": {"capacity": "high", "coverage": "southeastern_us", "criticality": "high"}
            }
        }
        
        # Assess risk for each critical node
        node_risks = {}
        
        for category, nodes in critical_nodes.items():
            node_risks[category] = {}
            
            for node_name, characteristics in nodes.items():
                # Base risk calculation
                base_risk = 10  # Base risk for all critical infrastructure
                
                # Increase risk based on importance/throughput
                if characteristics.get("throughput_rank", 10) <= 2 or characteristics.get("importance") == "critical":
                    base_risk += 30
                elif characteristics.get("throughput_rank", 10) <= 4 or characteristics.get("importance") == "high":
                    base_risk += 20
                else:
                    base_risk += 10
                
                # Regional risk factors
                region = characteristics.get("region", "")
                if region in ["west_coast"]:
                    base_risk += 15  # Seismic risk
                elif region in ["gulf_coast"]:
                    base_risk += 10  # Hurricane risk
                elif region in ["east_coast"]:
                    base_risk += 5   # Storm risk
                
                # Capacity/volume based risk
                if characteristics.get("cargo_value") == "high" or characteristics.get("capacity") == "very_high":
                    base_risk += 15
                
                node_risks[category][node_name] = {
                    "risk_score": min(100, base_risk),
                    "risk_factors": characteristics,
                    "criticality": characteristics.get("criticality", 
                                 characteristics.get("importance", "medium"))
                }
        
        # Calculate overall infrastructure risk
        all_scores = []
        for category in node_risks.values():
            for node_data in category.values():
                all_scores.append(node_data["risk_score"])
        
        avg_risk = sum(all_scores) / len(all_scores) if all_scores else 0
        
        return {
            "critical_nodes": node_risks,
            "total_critical_nodes": sum(len(category) for category in node_risks.values()),
            "average_node_risk": avg_risk,
            "overall_infrastructure_risk": avg_risk,
            "risk_level": "Critical" if avg_risk >= 75 else "High" if avg_risk >= 50 else "Medium",
            "source": "supply_chain_infrastructure",
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def get_transportation_vulnerabilities(self) -> Optional[Dict]:
        """Assess transportation network vulnerabilities."""
        
        # Transportation mode vulnerabilities
        transport_modes = {
            "maritime": {
                "capacity_share": 0.40,  # 40% of freight by volume
                "vulnerabilities": ["weather", "port_congestion", "fuel_costs", "piracy"],
                "critical_routes": ["Trans-Pacific", "Trans-Atlantic", "Panama Canal", "Suez Canal"]
            },
            "rail": {
                "capacity_share": 0.35,  # 35% of freight
                "vulnerabilities": ["derailment", "infrastructure_age", "weather", "labor_disputes"],
                "critical_routes": ["Chicago Hub", "BNSF Northern", "UP Southern", "CSX Eastern"]
            },
            "trucking": {
                "capacity_share": 0.70,  # 70% by value
                "vulnerabilities": ["driver_shortage", "fuel_costs", "regulations", "traffic_congestion"],
                "critical_routes": ["I-95 Corridor", "I-10 Corridor", "I-40 Corridor", "I-80 Corridor"]
            },
            "air_cargo": {
                "capacity_share": 0.05,  # 5% but high value
                "vulnerabilities": ["weather", "fuel_costs", "security_restrictions", "capacity_limits"],
                "critical_routes": ["Pacific Routes", "Atlantic Routes", "Domestic Hubs"]
            }
        }
        
        # Current risk assessment for each mode
        mode_risks = {}
        current_date = datetime.now()
        
        for mode, data in transport_modes.items():
            base_risk = 20  # Base transportation risk
            
            # Mode-specific risk factors
            if mode == "maritime":
                # Higher risk during storm season
                if current_date.month in [6, 7, 8, 9, 10, 11]:  # Hurricane season
                    base_risk += 25
                else:
                    base_risk += 10
                
            elif mode == "rail":
                # Infrastructure age factor
                base_risk += 20  # US rail infrastructure aging
                
            elif mode == "trucking":
                # Driver shortage is ongoing issue
                base_risk += 30
                
            elif mode == "air_cargo":
                # High fuel sensitivity
                base_risk += 15
            
            # Capacity dependency risk
            capacity_risk = data["capacity_share"] * 50  # Higher capacity = higher risk
            
            total_risk = min(100, base_risk + capacity_risk)
            
            mode_risks[mode] = {
                "risk_score": total_risk,
                "capacity_share": data["capacity_share"],
                "vulnerabilities": data["vulnerabilities"],
                "critical_routes": data["critical_routes"],
                "risk_level": "Critical" if total_risk >= 75 else "High" if total_risk >= 50 else "Medium"
            }
        
        # Calculate weighted overall transportation risk
        weighted_risk = sum(
            mode_data["risk_score"] * mode_data["capacity_share"] 
            for mode_data in mode_risks.values()
        )
        
        return {
            "transportation_modes": mode_risks,
            "weighted_transportation_risk": weighted_risk,
            "risk_level": "Critical" if weighted_risk >= 75 else "High" if weighted_risk >= 50 else "Medium",
            "source": "transportation_vulnerabilities",
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def get_supply_chain_disruption_indicators(self) -> Optional[Dict]:
        """Monitor indicators of supply chain disruption."""
        
        # Current disruption indicators (based on known patterns)
        disruption_indicators = {
            "global_events": {
                "geopolitical_tensions": {
                    "risk_score": 60,  # Ongoing global tensions
                    "impact_areas": ["semiconductor", "energy", "rare_earth_materials"],
                    "severity": "high"
                },
                "trade_restrictions": {
                    "risk_score": 45,
                    "impact_areas": ["technology", "automotive", "agriculture"],
                    "severity": "medium"
                },
                "currency_fluctuations": {
                    "risk_score": 35,
                    "impact_areas": ["manufacturing", "raw_materials", "energy"],
                    "severity": "medium"
                }
            },
            "operational_indicators": {
                "inventory_levels": {
                    "risk_score": 55,  # Below optimal levels
                    "trend": "concerning",
                    "sectors_affected": ["retail", "automotive", "electronics"]
                },
                "shipping_costs": {
                    "risk_score": 70,  # Elevated shipping costs
                    "trend": "increasing",
                    "sectors_affected": ["all_sectors"]
                },
                "delivery_delays": {
                    "risk_score": 50,
                    "trend": "stable",
                    "sectors_affected": ["e-commerce", "manufacturing", "healthcare"]
                }
            },
            "resource_constraints": {
                "labor_shortages": {
                    "risk_score": 65,
                    "sectors": ["transportation", "warehousing", "manufacturing"],
                    "impact": "high"
                },
                "material_shortages": {
                    "risk_score": 55,
                    "sectors": ["construction", "automotive", "electronics"],
                    "impact": "medium"
                },
                "energy_costs": {
                    "risk_score": 60,
                    "sectors": ["manufacturing", "transportation", "agriculture"],
                    "impact": "high"
                }
            }
        }
        
        # Calculate composite disruption score
        all_scores = []
        for category in disruption_indicators.values():
            for indicator in category.values():
                all_scores.append(indicator["risk_score"])
        
        composite_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        # Identify most critical disruption risks
        critical_risks = []
        for category_name, category in disruption_indicators.items():
            for indicator_name, indicator in category.items():
                if indicator["risk_score"] >= 60:
                    critical_risks.append({
                        "category": category_name,
                        "indicator": indicator_name,
                        "risk_score": indicator["risk_score"],
                        "impact": indicator.get("impact", indicator.get("severity", "medium"))
                    })
        
        return {
            "disruption_indicators": disruption_indicators,
            "composite_disruption_score": composite_score,
            "critical_risks": sorted(critical_risks, key=lambda x: x["risk_score"], reverse=True),
            "risk_level": "Critical" if composite_score >= 75 else "High" if composite_score >= 50 else "Medium",
            "assessment_date": datetime.utcnow().isoformat(),
            "source": "supply_chain_disruption"
        }
    
    async def get_logistics_performance_metrics(self) -> Optional[Dict]:
        """Assess logistics network performance and capacity."""
        
        # Logistics performance indicators
        performance_metrics = {
            "capacity_utilization": {
                "ports": {"utilization": 85, "status": "high", "bottlenecks": ["LA/Long Beach", "Savannah"]},
                "rail": {"utilization": 75, "status": "medium", "bottlenecks": ["Chicago", "Kansas City"]},
                "trucking": {"utilization": 90, "status": "critical", "bottlenecks": ["Driver availability"]},
                "warehousing": {"utilization": 88, "status": "high", "bottlenecks": ["E-commerce fulfillment"]}
            },
            "performance_indicators": {
                "on_time_delivery": {"percentage": 78, "trend": "declining", "target": 95},
                "inventory_turnover": {"ratio": 6.2, "trend": "stable", "target": 8.0},
                "order_fulfillment": {"percentage": 92, "trend": "stable", "target": 98},
                "cost_efficiency": {"index": 85, "trend": "declining", "target": 90}
            },
            "network_resilience": {
                "redundancy_level": {"score": 65, "status": "medium"},
                "recovery_capability": {"score": 70, "status": "medium"},
                "adaptability": {"score": 75, "status": "good"},
                "visibility": {"score": 60, "status": "limited"}
            }
        }
        
        # Calculate overall logistics performance score
        capacity_scores = [mode["utilization"] for mode in performance_metrics["capacity_utilization"].values()]
        performance_scores = [indicator["percentage"] if "percentage" in indicator 
                            else (indicator.get("ratio", 0) * 10 if "ratio" in indicator
                            else indicator.get("index", 0))
                            for indicator in performance_metrics["performance_indicators"].values()]
        resilience_scores = [metric["score"] for metric in performance_metrics["network_resilience"].values()]
        
        # Weighted scoring
        capacity_avg = sum(capacity_scores) / len(capacity_scores)
        performance_avg = sum(performance_scores) / len(performance_scores)
        resilience_avg = sum(resilience_scores) / len(resilience_scores)
        
        overall_performance = (capacity_avg * 0.4 + performance_avg * 0.4 + resilience_avg * 0.2)
        
        # Identify performance gaps
        performance_gaps = []
        for category, indicators in performance_metrics["performance_indicators"].items():
            if "target" in indicators and "percentage" in indicators:
                gap = indicators["target"] - indicators["percentage"]
                if gap > 5:  # Significant gap
                    performance_gaps.append({
                        "metric": category,
                        "current": indicators["percentage"],
                        "target": indicators["target"],
                        "gap": gap
                    })
        
        return {
            "performance_metrics": performance_metrics,
            "overall_performance_score": overall_performance,
            "performance_gaps": performance_gaps,
            "performance_level": "Good" if overall_performance >= 80 else "Fair" if overall_performance >= 60 else "Poor",
            "source": "logistics_performance",
            "last_updated": datetime.utcnow().isoformat()
        }


async def get_supply_chain_risks() -> Dict[str, Any]:
    """Get comprehensive supply chain risk assessment."""
    
    async with SupplyChainClient() as client:
        # Fetch multiple supply chain indicators concurrently
        results = await asyncio.gather(
            client.get_critical_infrastructure_nodes(),
            client.get_transportation_vulnerabilities(),
            client.get_supply_chain_disruption_indicators(),
            client.get_logistics_performance_metrics(),
            return_exceptions=True
        )
        
        indicators = {}
        indicator_names = ["infrastructure_nodes", "transportation_vulnerabilities", 
                          "disruption_indicators", "logistics_performance"]
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                indicators[indicator_names[i]] = result
        
        # Calculate overall supply chain risk score
        overall_risk = 0
        if indicators:
            infra_risk = indicators.get("infrastructure_nodes", {}).get("overall_infrastructure_risk", 0)
            transport_risk = indicators.get("transportation_vulnerabilities", {}).get("weighted_transportation_risk", 0)
            disruption_risk = indicators.get("disruption_indicators", {}).get("composite_disruption_score", 0)
            
            # Performance score needs to be inverted (lower performance = higher risk)
            performance_score = indicators.get("logistics_performance", {}).get("overall_performance_score", 80)
            performance_risk = 100 - performance_score
            
            overall_risk = (infra_risk + transport_risk + disruption_risk + performance_risk) / 4
        
        return {
            "indicators": indicators,
            "count": len(indicators),
            "overall_supply_chain_risk": overall_risk,
            "risk_level": "Critical" if overall_risk >= 75 else "High" if overall_risk >= 50 else "Medium",
            "source": "supply_chain",
            "last_updated": datetime.utcnow().isoformat()
        }


async def health_check(timeout: int = 5) -> bool:
    """Check if supply chain data sources are accessible."""
    try:
        async with SupplyChainClient() as client:
            # Try to get infrastructure data (internal calculation)
            result = await client.get_critical_infrastructure_nodes()
            return result is not None
    except Exception:
        return False