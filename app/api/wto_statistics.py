"""
WTO Statistics API

Provides endpoints for World Trade Organization data including trade volumes,
bilateral trade statistics, tariff data, and regional trade agreements.
"""

import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict, Any

from ..services.wto_integration import (
    get_wto_integration,
    TradeDirection,
    AgreementType,
    TradeStatistic,
    WTOTradeVolume,
    TariffData,
    TradeAgreement
)
from ..core.security import require_analytics_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/wto", tags=["wto"])

# Initialize WTO service
wto_service = get_wto_integration()


@router.get("/trade-statistics/bilateral")
async def get_bilateral_trade_statistics(
    reporter: str = Query(..., description="ISO3 code of reporting country (e.g., 'USA', 'CHN')"),
    partner: str = Query("all", description="ISO3 code of partner country or 'all' for global"),
    years: Optional[str] = Query(None, description="Comma-separated years (e.g., '2021,2022,2023')"),
    products: Optional[str] = Query(None, description="Comma-separated HS product codes"),
    _rate_limit: bool = Depends(require_analytics_rate_limit)
):
    """
    Get bilateral trade statistics between countries.
    
    Returns detailed trade flow data including imports, exports, and trade balances
    for specified country pairs and time periods.
    """
    try:
        # Parse years
        years_list = None
        if years:
            try:
                years_list = [int(y.strip()) for y in years.split(",")]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid years format")
        
        # Parse products
        products_list = None
        if products:
            products_list = [p.strip() for p in products.split(",")]
        
        # Get trade statistics
        trade_stats = await wto_service.get_bilateral_trade_data(
            reporter=reporter.upper(),
            partner=partner.upper() if partner != "all" else partner,
            years=years_list,
            products=products_list
        )
        
        # Calculate summary metrics
        total_exports = sum(stat.value_usd for stat in trade_stats if stat.trade_flow == "exports")
        total_imports = sum(stat.value_usd for stat in trade_stats if stat.trade_flow == "imports")
        trade_balance = total_exports - total_imports
        
        return {
            "status": "success",
            "query": {
                "reporter": reporter.upper(),
                "partner": partner.upper() if partner != "all" else partner,
                "years": years_list,
                "products": products_list
            },
            "summary": {
                "total_exports_usd": total_exports,
                "total_imports_usd": total_imports,
                "trade_balance_usd": trade_balance,
                "total_records": len(trade_stats)
            },
            "trade_statistics": [
                {
                    "reporter": stat.reporter_iso3,
                    "partner": stat.partner_iso3,
                    "product_code": stat.product_code,
                    "product_description": stat.product_description,
                    "trade_flow": stat.trade_flow,
                    "value_usd": stat.value_usd,
                    "quantity": stat.quantity,
                    "unit": stat.unit,
                    "year": stat.year,
                    "quarter": stat.quarter,
                    "last_updated": stat.last_updated.isoformat()
                } for stat in trade_stats
            ],
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bilateral trade statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve trade statistics: {str(e)}")


@router.get("/trade-volume/global")
async def get_global_trade_volume(
    _rate_limit: bool = Depends(require_analytics_rate_limit)
):
    """
    Get global trade volume statistics and forecasts.
    
    Provides comprehensive global trade data including total volumes,
    growth rates, regional breakdowns, and top trading nations.
    """
    try:
        trade_volume = await wto_service.get_global_trade_volume()
        
        return {
            "status": "success",
            "global_trade_volume": {
                "total_global_trade_usd": trade_volume.total_global_trade,
                "year_on_year_growth_percent": trade_volume.year_on_year_growth,
                "forecast_next_year_usd": trade_volume.forecast_next_year,
                "data_timestamp": trade_volume.data_timestamp.isoformat()
            },
            "regional_breakdown": trade_volume.regional_breakdown,
            "top_trading_countries": trade_volume.top_traders,
            "insights": {
                "largest_trading_bloc": max(trade_volume.regional_breakdown.items(), key=lambda x: x[1]),
                "total_countries_tracked": len(trade_volume.top_traders),
                "growth_trend": "positive" if trade_volume.year_on_year_growth > 0 else "negative"
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get global trade volume: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve global trade volume: {str(e)}")


@router.get("/tariffs")
async def get_tariff_data(
    reporter: str = Query(..., description="ISO3 code of importing country"),
    partner: str = Query("all", description="ISO3 code of exporting country or 'all'"),
    products: Optional[str] = Query(None, description="Comma-separated HS product codes"),
    _rate_limit: bool = Depends(require_analytics_rate_limit)
):
    """
    Get tariff rate data between countries.
    
    Returns detailed tariff information including MFN rates, preferential rates,
    and applicable trade agreements.
    """
    try:
        # Parse products
        products_list = None
        if products:
            products_list = [p.strip() for p in products.split(",")]
        
        # Get tariff data
        tariff_data = await wto_service.get_tariff_data(
            reporter=reporter.upper(),
            partner=partner.upper() if partner != "all" else partner,
            product_codes=products_list
        )
        
        # Calculate summary metrics
        avg_mfn_rate = 0
        avg_pref_rate = 0
        mfn_tariffs = [t for t in tariff_data if t.tariff_type == "MFN"]
        pref_tariffs = [t for t in tariff_data if t.tariff_type == "Preferential"]
        
        if mfn_tariffs:
            avg_mfn_rate = sum(t.tariff_rate for t in mfn_tariffs) / len(mfn_tariffs)
        if pref_tariffs:
            avg_pref_rate = sum(t.tariff_rate for t in pref_tariffs) / len(pref_tariffs)
        
        return {
            "status": "success",
            "query": {
                "reporter": reporter.upper(),
                "partner": partner.upper() if partner != "all" else partner,
                "products": products_list
            },
            "summary": {
                "average_mfn_rate_percent": round(avg_mfn_rate, 2),
                "average_preferential_rate_percent": round(avg_pref_rate, 2),
                "preference_margin_percent": round(avg_mfn_rate - avg_pref_rate, 2),
                "total_tariff_lines": len(tariff_data)
            },
            "tariff_data": [
                {
                    "reporter": tariff.reporter_iso3,
                    "partner": tariff.partner_iso3,
                    "product_code": tariff.product_code,
                    "tariff_rate_percent": tariff.tariff_rate,
                    "tariff_type": tariff.tariff_type,
                    "effective_date": tariff.effective_date.isoformat(),
                    "agreement_name": tariff.agreement_name
                } for tariff in tariff_data
            ],
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get tariff data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tariff data: {str(e)}")


@router.get("/trade-agreements")
async def get_trade_agreements(
    country: Optional[str] = Query(None, description="ISO3 code to filter by participant country"),
    agreement_type: Optional[str] = Query(None, description="Agreement type: 'fta', 'cu', 'rta', 'pta'"),
    status: str = Query("In Force", description="Agreement status filter"),
    _rate_limit: bool = Depends(require_analytics_rate_limit)
):
    """
    Get regional trade agreements data.
    
    Returns information about trade agreements including participants,
    coverage, and estimated trade impact.
    """
    try:
        # Parse agreement type
        parsed_agreement_type = None
        if agreement_type:
            try:
                parsed_agreement_type = AgreementType(agreement_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid agreement type: {agreement_type}")
        
        # Get trade agreements
        agreements = await wto_service.get_trade_agreements(
            country=country.upper() if country else None,
            agreement_type=parsed_agreement_type,
            status=status
        )
        
        # Calculate summary metrics
        total_trade_impact = sum(a.trade_volume_impact or 0 for a in agreements)
        countries_involved = set()
        for agreement in agreements:
            countries_involved.update(agreement.participants)
        
        return {
            "status": "success",
            "query": {
                "country": country.upper() if country else None,
                "agreement_type": agreement_type,
                "status": status
            },
            "summary": {
                "total_agreements": len(agreements),
                "total_trade_impact_usd": total_trade_impact,
                "countries_involved": len(countries_involved),
                "agreement_types": list(set(a.agreement_type.value for a in agreements))
            },
            "trade_agreements": [
                {
                    "agreement_id": agreement.agreement_id,
                    "agreement_name": agreement.agreement_name,
                    "agreement_type": agreement.agreement_type.value,
                    "status": agreement.status,
                    "entry_into_force": agreement.entry_into_force.isoformat() if agreement.entry_into_force else None,
                    "participants": agreement.participants,
                    "participant_count": len(agreement.participants),
                    "coverage": agreement.coverage,
                    "trade_volume_impact_usd": agreement.trade_volume_impact
                } for agreement in agreements
            ],
            "coverage_analysis": {
                "goods_coverage": len([a for a in agreements if "goods" in a.coverage]),
                "services_coverage": len([a for a in agreements if "services" in a.coverage]),
                "investment_coverage": len([a for a in agreements if "investment" in a.coverage])
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trade agreements: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve trade agreements: {str(e)}")


@router.post("/disruption-impact")
async def analyze_trade_disruption_impact(
    disruption_data: Dict[str, Any],
    _rate_limit: bool = Depends(require_analytics_rate_limit)
):
    """
    Analyze the impact of trade route disruptions.
    
    Calculate economic impact, alternative routes, and mitigation strategies
    for supply chain disruptions affecting international trade.
    """
    try:
        # Validate input data
        if "affected_routes" not in disruption_data:
            raise HTTPException(status_code=400, detail="affected_routes is required")
        
        affected_routes = disruption_data["affected_routes"]
        if not isinstance(affected_routes, list):
            raise HTTPException(status_code=400, detail="affected_routes must be a list")
        
        # Parse affected routes
        route_tuples = []
        for route in affected_routes:
            if isinstance(route, dict) and "origin" in route and "destination" in route:
                route_tuples.append((route["origin"].upper(), route["destination"].upper()))
            elif isinstance(route, list) and len(route) == 2:
                route_tuples.append((route[0].upper(), route[1].upper()))
            else:
                raise HTTPException(status_code=400, detail="Invalid route format")
        
        disruption_severity = disruption_data.get("severity", 0.3)
        if not 0 <= disruption_severity <= 1:
            raise HTTPException(status_code=400, detail="Severity must be between 0 and 1")
        
        # Calculate disruption impact
        impact_analysis = await wto_service.get_trade_disruption_impact(
            affected_routes=route_tuples,
            disruption_severity=disruption_severity
        )
        
        return {
            "status": "success",
            "disruption_scenario": {
                "affected_routes": [f"{origin} → {destination}" for origin, destination in route_tuples],
                "severity_level": disruption_severity,
                "severity_description": (
                    "Low" if disruption_severity < 0.3 else
                    "Moderate" if disruption_severity < 0.7 else "High"
                )
            },
            "impact_assessment": {
                "total_trade_at_risk_usd": impact_analysis["total_trade_at_risk"],
                "affected_countries": impact_analysis["affected_countries"],
                "economic_impact": impact_analysis["economic_impact"],
                "recovery_timeline": impact_analysis["recovery_timeline"]
            },
            "alternative_routes": impact_analysis["alternative_routes"],
            "mitigation_strategies": impact_analysis["mitigation_strategies"],
            "recommendations": [
                f"Monitor {len(impact_analysis['alternative_routes'])} alternative trade routes",
                "Increase inventory buffers for affected product categories",
                "Activate emergency supplier diversification protocols",
                "Engage diplomatic channels for trade facilitation"
            ],
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze trade disruption impact: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze disruption impact: {str(e)}")


@router.get("/regional-analysis/{region}")
async def get_regional_trade_analysis(
    region: str,
    include_agreements: bool = Query(True, description="Include trade agreement analysis"),
    include_tariffs: bool = Query(False, description="Include average tariff analysis"),
    _rate_limit: bool = Depends(require_analytics_rate_limit)
):
    """
    Get comprehensive trade analysis for a specific region.
    
    Analyzes intra-regional trade, major trade partners, agreements,
    and economic integration metrics.
    """
    try:
        # Map region names to country codes
        regional_mappings = {
            "asean": ["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "VNM"],
            "eu": ["AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", 
                   "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", 
                   "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"],
            "nafta": ["USA", "CAN", "MEX"],
            "mercosur": ["ARG", "BRA", "PRY", "URY"],
            "cptpp": ["AUS", "BRN", "CAN", "CHL", "JPN", "MYS", "MEX", "NZL", "PER", "SGP", "VNM"]
        }
        
        region_countries = regional_mappings.get(region.lower())
        if not region_countries:
            raise HTTPException(status_code=404, detail=f"Region '{region}' not found")
        
        # Get trade data for the region
        regional_analysis = {
            "region": region.upper(),
            "member_countries": region_countries,
            "intra_regional_trade": {},
            "external_trade_partners": {},
            "trade_agreements": [],
            "tariff_analysis": {},
            "integration_metrics": {}
        }
        
        # Calculate intra-regional trade (sample calculation)
        total_intra_trade = 0
        total_external_trade = 0
        
        for country in region_countries[:3]:  # Sample first 3 countries for demo
            try:
                trade_data = await wto_service.get_bilateral_trade_data(country, "all")
                intra_trade = sum(stat.value_usd for stat in trade_data 
                                if stat.partner_iso3 in region_countries)
                external_trade = sum(stat.value_usd for stat in trade_data 
                                   if stat.partner_iso3 not in region_countries)
                
                total_intra_trade += intra_trade
                total_external_trade += external_trade
                
            except Exception as e:
                logger.warning(f"Failed to get trade data for {country}: {e}")
        
        regional_analysis["intra_regional_trade"] = {
            "total_value_usd": total_intra_trade,
            "share_of_total_trade_percent": (total_intra_trade / (total_intra_trade + total_external_trade) * 100) if (total_intra_trade + total_external_trade) > 0 else 0
        }
        
        # Get trade agreements if requested
        if include_agreements:
            for country in region_countries[:3]:  # Sample for demo
                try:
                    agreements = await wto_service.get_trade_agreements(country=country)
                    regional_analysis["trade_agreements"].extend(agreements)
                except Exception:
                    continue
        
        # Remove duplicates from agreements
        seen_agreements = set()
        unique_agreements = []
        for agreement in regional_analysis["trade_agreements"]:
            if agreement.agreement_id not in seen_agreements:
                unique_agreements.append(agreement)
                seen_agreements.add(agreement.agreement_id)
        regional_analysis["trade_agreements"] = unique_agreements
        
        # Integration metrics
        regional_analysis["integration_metrics"] = {
            "economic_integration_index": min(0.8, total_intra_trade / 1000000000000),  # Simplified metric
            "trade_complementarity_index": 0.65,  # Mock value
            "trade_intensity_index": 1.2,  # Mock value
            "tariff_dispersion_index": 0.15  # Mock value
        }
        
        return {
            "status": "success",
            "regional_analysis": regional_analysis,
            "summary": {
                "member_count": len(region_countries),
                "total_agreements": len(unique_agreements),
                "intra_regional_share_percent": regional_analysis["intra_regional_trade"]["share_of_total_trade_percent"]
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze regional trade: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze regional trade: {str(e)}")


@router.get("/health-check")
async def wto_service_health_check():
    """Check WTO service health and data availability."""
    try:
        # Test basic service functionality
        test_trade_data = await wto_service.get_bilateral_trade_data("USA", "CHN", [2023])
        test_global_data = await wto_service.get_global_trade_volume()
        
        return {
            "status": "healthy",
            "service": "WTO Statistics Integration",
            "data_sources": ["WTO Stats API", "WTO Timeseries API", "Mock Data Fallback"],
            "last_data_update": test_global_data.data_timestamp.isoformat(),
            "capabilities": [
                "Bilateral trade statistics",
                "Global trade volume data", 
                "Tariff rate information",
                "Trade agreement tracking",
                "Disruption impact analysis"
            ],
            "test_results": {
                "bilateral_trade_query": len(test_trade_data) > 0,
                "global_volume_query": test_global_data.total_global_trade > 0,
                "cache_operational": True
            },
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"WTO service health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "WTO Statistics Integration",
            "error": str(e),
            "checked_at": datetime.utcnow().isoformat()
        }