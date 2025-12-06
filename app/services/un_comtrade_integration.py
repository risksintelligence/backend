"""
UN Comtrade API Integration for Global Trade Statistics
Uses official comtradeapicall library with proper authentication
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from app.core.cache import cache_with_fallback, CacheConfig
from app.core.config import settings

# Import official UN Comtrade library
try:
    import comtradeapicall
except ImportError:
    comtradeapicall = None
    logger.warning("comtradeapicall library not installed")

logger = logging.getLogger(__name__)

class UNComtradeIntegration:
    """UN Comtrade API integration using official comtradeapicall library"""
    
    def __init__(self):
        self.primary_key = settings.comtrade_primary_key
        self.secondary_key = settings.comtrade_secondary_key
        self.current_key = self.primary_key  # Start with primary key
    
    def _get_api_key(self) -> Optional[str]:
        """Get current API key"""
        return self.current_key if self.current_key else None
    
    def _switch_api_key(self):
        """Switch to secondary key if primary fails"""
        if self.current_key == self.primary_key and self.secondary_key:
            self.current_key = self.secondary_key
            logger.info("Switched to secondary Comtrade API key")
            return True
        return False
    
    async def get_trade_data(self, 
                           reporter_code: str = "840",  # USA
                           partner_code: str = "156",   # China
                           time_period: str = "2021",
                           trade_flow: str = "M",       # Imports (M=Import, X=Export)
                           commodity_code: str = "TOTAL") -> Optional[pd.DataFrame]:
        """Get bilateral trade data using official comtradeapicall library"""
        
        if not comtradeapicall:
            logger.error("comtradeapicall library not available")
            return None
            
        api_key = self._get_api_key()
        if not api_key:
            logger.error("No Comtrade API key available")
            return None
        
        try:
            # Use official library with simplified approach
            # Try authenticated call first
            if api_key:
                try:
                    df = comtradeapicall.getFinalData(
                        subscription_key=api_key,
                        typeCode='C',           # Commodities
                        freqCode='A',           # Annual
                        clCode='HS',            # Harmonized System
                        period=time_period,
                        reporterCode=reporter_code,
                        cmdCode=commodity_code,
                        flowCode=trade_flow,
                        partnerCode=partner_code,
                        partner2Code='0',       # No secondary partner
                        customsCode='C00',      # General customs procedure
                        motCode='0',            # All modes of transport
                        maxRecords=500,         # Limit records
                        format_output='JSON',   # JSON output format
                        aggregateBy=None,       # No aggregation
                        breakdownMode='classic', # Classic breakdown
                        countOnly=False,        # Return data, not count
                        includeDesc=True        # Include descriptions
                    )
                    if df is not None and not df.empty:
                        logger.info(f"Retrieved {len(df)} trade records from UN Comtrade (authenticated)")
                        return df
                except Exception as e:
                    logger.warning(f"Authenticated call failed, trying preview: {e}")
            
            # Fallback to preview data (limited but no auth required)
            df = comtradeapicall.previewFinalData(
                typeCode='C',           # Commodities
                freqCode='A',           # Annual
                clCode='HS',            # Harmonized System
                period=time_period,
                reporterCode=reporter_code,
                cmdCode=commodity_code,
                flowCode=trade_flow,
                partnerCode=partner_code,
                partner2Code='0',       # No secondary partner
                customsCode='C00',      # General customs procedure
                motCode='0',            # All modes of transport
                maxRecords=100,         # Lower limit for preview
                format_output='JSON',   # JSON output format
                aggregateBy=None,       # No aggregation
                breakdownMode='classic', # Classic breakdown
                countOnly=False,        # Return data, not count
                includeDesc=True        # Include descriptions
            )
            
            if df is not None and not df.empty:
                logger.info(f"Retrieved {len(df)} trade records from UN Comtrade")
                return df
            else:
                logger.warning("No trade data returned from UN Comtrade")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch trade data: {e}")
            # Try switching API key if request failed
            if self._switch_api_key():
                return await self.get_trade_data(reporter_code, partner_code, time_period, trade_flow, commodity_code)
            return None
    
    async def build_supply_chain_network(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build supply chain network nodes and edges from trade data"""
        try:
            trade_matrix = await self.get_global_trade_matrix()
            
            # Create nodes (countries)
            nodes = []
            countries = {
                "840": {"name": "USA", "region": "North America"},
                "156": {"name": "China", "region": "Asia"},
                "276": {"name": "Germany", "region": "Europe"},
                "392": {"name": "Japan", "region": "Asia"},
                "826": {"name": "UK", "region": "Europe"},
                "250": {"name": "France", "region": "Europe"},
                "356": {"name": "India", "region": "Asia"},
                "380": {"name": "Italy", "region": "Europe"}
            }
            
            for code, info in countries.items():
                node = {
                    "id": code,
                    "name": info["name"],
                    "region": info["region"],
                    "trade_volume": 0,
                    "import_dependence": 0.0,
                    "export_dependence": 0.0,
                    "type": "country"
                }
                nodes.append(node)
            
            # Create edges (trade relationships)
            edges = []
            trade_flows = trade_matrix.get("trade_flows", {})
            
            for flow_key, flow_data in trade_flows.items():
                # Parse country pair from flow_key (e.g., "USA-CHN")
                if "-" in flow_key:
                    countries_pair = flow_key.split("-")
                    if len(countries_pair) == 2:
                        source_country = countries_pair[0]
                        target_country = countries_pair[1]
                        
                        # Find country codes for these names
                        source_code = None
                        target_code = None
                        for code, info in countries.items():
                            if info["name"] == source_country or source_country in info["name"]:
                                source_code = code
                            if info["name"] == target_country or target_country in info["name"]:
                                target_code = code
                        
                        if source_code and target_code:
                            imports_value = flow_data.get("imports", {}).get("value", 0)
                            exports_value = flow_data.get("exports", {}).get("value", 0)
                            
                            # Create edge for imports (target imports from source)
                            if imports_value > 0:
                                edge = {
                                    "source": source_code,
                                    "target": target_code,
                                    "type": "import",
                                    "value": imports_value,
                                    "weight": min(imports_value / 1e9, 100),  # Normalize to reasonable weight
                                    "risk_level": "medium" if imports_value > 100e9 else "low"
                                }
                                edges.append(edge)
                            
                            # Create edge for exports (source exports to target)
                            if exports_value > 0:
                                edge = {
                                    "source": target_code,
                                    "target": source_code,
                                    "type": "export",
                                    "value": exports_value,
                                    "weight": min(exports_value / 1e9, 100),  # Normalize to reasonable weight
                                    "risk_level": "medium" if exports_value > 100e9 else "low"
                                }
                                edges.append(edge)
            
            # Update node trade volumes based on edges
            for node in nodes:
                node_id = node["id"]
                total_imports = sum(edge["value"] for edge in edges if edge["target"] == node_id and edge["type"] == "import")
                total_exports = sum(edge["value"] for edge in edges if edge["source"] == node_id and edge["type"] == "export")
                node["trade_volume"] = total_imports + total_exports
                
                # Calculate dependence metrics
                if total_imports + total_exports > 0:
                    node["import_dependence"] = total_imports / (total_imports + total_exports)
                    node["export_dependence"] = total_exports / (total_imports + total_exports)
            
            logger.info(f"Built supply chain network: {len(nodes)} nodes, {len(edges)} edges")
            return nodes, edges
            
        except Exception as e:
            logger.error(f"Failed to build supply chain network: {e}")
            # Return empty network on error
            return [], []
    
    async def get_global_trade_matrix(self) -> Dict[str, Any]:
        """Get trade matrix for major economies using official API"""
        try:
            # Major economies with UN country codes
            countries = {
                "840": "USA",      # United States
                "156": "CHN",      # China
                "276": "DEU",      # Germany
                "392": "JPN",      # Japan
                "826": "GBR",      # United Kingdom
                "250": "FRA",      # France
                "356": "IND",      # India
                "380": "ITA",      # Italy
            }
            
            trade_flows = {}
            trade_summary = {"total_imports": 0, "total_exports": 0, "trade_balance": {}}
            
            # Get trade data for key bilateral relationships
            key_pairs = [
                ("840", "156"),  # USA-China
                ("840", "276"),  # USA-Germany
                ("156", "276"),  # China-Germany
                ("392", "156"),  # Japan-China
                ("826", "276"),  # UK-Germany
            ]
            
            for reporter, partner in key_pairs:
                try:
                    # Get imports
                    imports_df = await self.get_trade_data(
                        reporter_code=reporter,
                        partner_code=partner,
                        trade_flow="M"  # Imports
                    )
                    
                    # Get exports  
                    exports_df = await self.get_trade_data(
                        reporter_code=reporter,
                        partner_code=partner,
                        trade_flow="X"  # Exports
                    )
                    
                    pair_key = f"{countries.get(reporter, reporter)}-{countries.get(partner, partner)}"
                    
                    imports_value = self._calculate_total_value(imports_df) if imports_df is not None else 0
                    exports_value = self._calculate_total_value(exports_df) if exports_df is not None else 0
                    
                    trade_flows[pair_key] = {
                        "imports": {
                            "value": imports_value,
                            "records": len(imports_df) if imports_df is not None else 0
                        },
                        "exports": {
                            "value": exports_value,
                            "records": len(exports_df) if exports_df is not None else 0
                        },
                        "last_updated": datetime.utcnow().isoformat()
                    }
                    
                    # Update summary
                    trade_summary["total_imports"] += imports_value
                    trade_summary["total_exports"] += exports_value
                    trade_summary["trade_balance"][pair_key] = exports_value - imports_value
                    
                    # Rate limiting - be respectful to the API
                    await asyncio.sleep(1.0)
                    
                except Exception as e:
                    logger.warning(f"Failed to get trade data for {reporter}-{partner}: {e}")
                    continue
            
            # Calculate trade concentration and vulnerability metrics
            trade_metrics = self._calculate_trade_metrics(trade_flows, trade_summary)
            
            return {
                "trade_flows": trade_flows,
                "summary": trade_summary,
                "metrics": trade_metrics,
                "countries_analyzed": len(countries),
                "data_source": "UN Comtrade (Official API)",
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get global trade matrix: {e}")
            return self._get_fallback_trade_matrix()
    
    def _calculate_total_value(self, df: pd.DataFrame) -> float:
        """Calculate total trade value from DataFrame"""
        if df is None or df.empty:
            return 0.0
            
        # Look for common trade value columns
        value_columns = ['primaryValue', 'tradeValue', 'value', 'TradeValue']
        
        for col in value_columns:
            if col in df.columns:
                return float(df[col].sum())
        
        logger.warning("No recognized value column found in trade data")
        return 0.0
    
    def _calculate_trade_metrics(self, trade_flows: Dict, summary: Dict) -> Dict[str, Any]:
        """Calculate trade concentration and risk metrics"""
        try:
            metrics = {
                "trade_concentration": 0.0,
                "supply_chain_diversity": 0.0,
                "trade_vulnerability_score": 0.0,
                "largest_trade_relationship": {},
                "trade_growth_estimate": 0.0
            }
            
            if not trade_flows:
                return metrics
            
            # Calculate trade concentration (Herfindahl index approximation)
            total_trade = summary.get("total_imports", 0) + summary.get("total_exports", 0)
            if total_trade > 0:
                concentration = 0
                for flow_data in trade_flows.values():
                    flow_total = flow_data["imports"]["value"] + flow_data["exports"]["value"]
                    share = (flow_total / total_trade) ** 2
                    concentration += share
                
                metrics["trade_concentration"] = round(concentration, 3)
                metrics["supply_chain_diversity"] = round(1 - concentration, 3)
            
            # Find largest trade relationship
            largest_value = 0
            largest_pair = ""
            for pair, data in trade_flows.items():
                total_value = data["imports"]["value"] + data["exports"]["value"]
                if total_value > largest_value:
                    largest_value = total_value
                    largest_pair = pair
            
            if largest_pair:
                metrics["largest_trade_relationship"] = {
                    "pair": largest_pair,
                    "value": largest_value,
                    "share_of_total": round(largest_value / total_trade * 100, 1) if total_trade > 0 else 0
                }
            
            # Calculate vulnerability score (higher concentration = higher vulnerability)
            vulnerability = 0
            if metrics["trade_concentration"] > 0.3:
                vulnerability += 30
            elif metrics["trade_concentration"] > 0.2:
                vulnerability += 20
            elif metrics["trade_concentration"] > 0.1:
                vulnerability += 10
            
            # Add balance factors
            balances = summary.get("trade_balance", {})
            extreme_imbalances = sum(1 for balance in balances.values() if abs(balance) > 100000000000)  # >$100B
            vulnerability += min(extreme_imbalances * 10, 30)
            
            metrics["trade_vulnerability_score"] = min(vulnerability, 100)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate trade metrics: {e}")
            return {
                "trade_concentration": 0.0,
                "supply_chain_diversity": 0.0,
                "trade_vulnerability_score": 50.0,
                "largest_trade_relationship": {},
                "trade_growth_estimate": 0.0
            }
    
    def _get_fallback_trade_matrix(self) -> Dict[str, Any]:
        """Return fallback trade data when API fails"""
        logger.warning("Using fallback UN Comtrade data")
        return {
            "trade_flows": {
                "USA-CHN": {
                    "imports": {"value": 506400000000, "records": 50},
                    "exports": {"value": 151800000000, "records": 45}
                },
                "USA-DEU": {
                    "imports": {"value": 156100000000, "records": 35},
                    "exports": {"value": 64900000000, "records": 30}
                },
                "CHN-DEU": {
                    "imports": {"value": 191200000000, "records": 40},
                    "exports": {"value": 107400000000, "records": 38}
                }
            },
            "summary": {
                "total_imports": 853700000000,
                "total_exports": 324100000000,
                "trade_balance": {
                    "USA-CHN": -354600000000,
                    "USA-DEU": -91200000000,
                    "CHN-DEU": 83800000000
                }
            },
            "metrics": {
                "trade_concentration": 0.235,
                "supply_chain_diversity": 0.765,
                "trade_vulnerability_score": 45,
                "largest_trade_relationship": {
                    "pair": "USA-CHN",
                    "value": 658200000000,
                    "share_of_total": 55.9
                }
            },
            "countries_analyzed": 8,
            "data_source": "UN Comtrade (fallback)",
            "last_updated": datetime.utcnow().isoformat()
        }

# Global instance
un_comtrade = UNComtradeIntegration()

@cache_with_fallback(
    config=CacheConfig(
        key_prefix="un_comtrade_global",
        ttl_seconds=7200,  # 2 hour cache (API rate limits)
        fallback_ttl_seconds=86400  # 24 hour fallback
    )
)
async def get_global_trade_statistics() -> Dict[str, Any]:
    """Get comprehensive global trade statistics using official API"""
    try:
        return await un_comtrade.get_global_trade_matrix()
    except Exception as e:
        logger.error(f"UN Comtrade integration failed: {e}")
        return un_comtrade._get_fallback_trade_matrix()

@cache_with_fallback(
    config=CacheConfig(
        key_prefix="un_comtrade_bilateral",
        ttl_seconds=3600,  # 1 hour cache
        fallback_ttl_seconds=43200  # 12 hour fallback
    )
)
async def get_bilateral_trade(reporter: str = "840", partner: str = "156") -> Dict[str, Any]:
    """Get detailed bilateral trade analysis using official API"""
    try:
        imports_df = await un_comtrade.get_trade_data(
            reporter_code=reporter, 
            partner_code=partner, 
            trade_flow="M"  # Imports
        )
        exports_df = await un_comtrade.get_trade_data(
            reporter_code=reporter, 
            partner_code=partner, 
            trade_flow="X"  # Exports
        )
        
        imports_value = un_comtrade._calculate_total_value(imports_df) if imports_df is not None else 0
        exports_value = un_comtrade._calculate_total_value(exports_df) if exports_df is not None else 0
        
        return {
            "imports": {
                "value": imports_value,
                "records": len(imports_df) if imports_df is not None else 0
            },
            "exports": {
                "value": exports_value,
                "records": len(exports_df) if exports_df is not None else 0
            },
            "reporter": reporter,
            "partner": partner,
            "data_source": "UN Comtrade (Official API)",
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Bilateral trade analysis failed: {e}")
        return {
            "imports": {"value": 0, "records": 0},
            "exports": {"value": 0, "records": 0},
            "reporter": reporter,
            "partner": partner,
            "data_source": "UN Comtrade (fallback)",
            "last_updated": datetime.utcnow().isoformat()
        }

async def cleanup_comtrade():
    """Cleanup function (no persistent connections in official library)"""
    logger.info("UN Comtrade cleanup completed")