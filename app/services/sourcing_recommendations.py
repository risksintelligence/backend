"""
Alternative Sourcing Recommendation Engine

Provides intelligent recommendations for alternative suppliers, trade routes, and sourcing
strategies based on supply chain disruptions, risk assessments, and historical patterns.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import math

logger = logging.getLogger(__name__)


class SourcingPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"


class RecommendationType(Enum):
    SUPPLIER_DIVERSIFICATION = "supplier_diversification"
    ROUTE_OPTIMIZATION = "route_optimization"
    INVENTORY_BUFFERING = "inventory_buffering"
    NEARSHORING = "nearshoring"
    VERTICAL_INTEGRATION = "vertical_integration"
    CONTRACT_HEDGING = "contract_hedging"


@dataclass
class SupplierProfile:
    supplier_id: str
    supplier_name: str
    country: str
    region: str
    coordinates: Tuple[float, float]  # (lat, lng)
    capacity_rating: float  # 0.0 to 1.0
    reliability_score: float  # 0.0 to 1.0  
    cost_index: float  # Relative cost (1.0 = baseline)
    quality_score: float  # 0.0 to 1.0
    lead_time_days: int
    certifications: List[str]
    specialties: List[str]
    risk_factors: Dict[str, float]
    last_updated: datetime


@dataclass
class TradeRoute:
    route_id: str
    route_name: str
    origin_country: str
    destination_country: str
    transit_countries: List[str]
    transport_modes: List[str]  # sea, air, rail, road
    average_transit_days: int
    cost_factor: float  # Relative to baseline route
    reliability_score: float  # 0.0 to 1.0
    risk_score: float  # 0.0 to 1.0
    capacity_utilization: float  # 0.0 to 1.0
    seasonal_factors: Dict[str, float]  # Q1, Q2, Q3, Q4 multipliers


@dataclass
class SourcingRecommendation:
    recommendation_id: str
    recommendation_type: RecommendationType
    priority: SourcingPriority
    title: str
    description: str
    affected_commodities: List[str]
    target_regions: List[str]
    implementation_timeline_weeks: int
    cost_impact_percent: float  # Positive = cost increase, negative = savings
    risk_reduction_percent: float
    confidence_score: float  # 0.0 to 1.0
    alternative_suppliers: List[SupplierProfile]
    alternative_routes: List[TradeRoute]
    implementation_steps: List[str]
    success_metrics: List[str]
    dependencies: List[str]
    created_at: datetime


class SourcingRecommendationEngine:
    """Intelligent sourcing recommendation engine for supply chain optimization."""
    
    def __init__(self):
        # Load supplier database
        self.suppliers_db = self._initialize_suppliers_database()
        
        # Load trade routes database  
        self.routes_db = self._initialize_routes_database()
        
        # Regional risk factors
        self.regional_risks = {
            "east_asia": 0.35,
            "southeast_asia": 0.28,
            "middle_east": 0.62,
            "eastern_europe": 0.45,
            "north_america": 0.18,
            "western_europe": 0.22,
            "latin_america": 0.38,
            "africa": 0.55,
            "oceania": 0.15
        }
        
        # Commodity-specific sourcing strategies
        self.commodity_strategies = {
            "semiconductors": {
                "critical_suppliers": ["taiwan", "south_korea", "japan"],
                "lead_time_sensitivity": 0.9,
                "quality_importance": 0.95,
                "diversification_priority": "critical"
            },
            "rare_earth_metals": {
                "critical_suppliers": ["china", "australia", "myanmar"],
                "lead_time_sensitivity": 0.6,
                "quality_importance": 0.85,
                "diversification_priority": "critical"
            },
            "crude_oil": {
                "critical_suppliers": ["saudi_arabia", "russia", "united_states"],
                "lead_time_sensitivity": 0.8,
                "quality_importance": 0.70,
                "diversification_priority": "high"
            },
            "manufactured_goods": {
                "critical_suppliers": ["china", "germany", "united_states"],
                "lead_time_sensitivity": 0.7,
                "quality_importance": 0.75,
                "diversification_priority": "medium"
            }
        }

    def _initialize_suppliers_database(self) -> List[SupplierProfile]:
        """Initialize comprehensive supplier database."""
        suppliers = [
            # Technology/Semiconductor Suppliers
            SupplierProfile(
                supplier_id="tsmc_taiwan",
                supplier_name="Taiwan Semiconductor Manufacturing",
                country="taiwan",
                region="east_asia",
                coordinates=(25.0330, 121.5654),
                capacity_rating=0.95,
                reliability_score=0.92,
                cost_index=1.15,
                quality_score=0.98,
                lead_time_days=90,
                certifications=["ISO9001", "ISO14001", "IATF16949"],
                specialties=["semiconductors", "advanced_chips", "ai_processors"],
                risk_factors={"geopolitical": 0.65, "natural_disaster": 0.40, "supply_disruption": 0.25},
                last_updated=datetime.utcnow()
            ),
            SupplierProfile(
                supplier_id="samsung_korea", 
                supplier_name="Samsung Electronics",
                country="south_korea",
                region="east_asia", 
                coordinates=(37.5665, 126.9780),
                capacity_rating=0.88,
                reliability_score=0.85,
                cost_index=1.08,
                quality_score=0.94,
                lead_time_days=75,
                certifications=["ISO9001", "ISO14001", "OHSAS18001"],
                specialties=["semiconductors", "memory_chips", "displays"],
                risk_factors={"geopolitical": 0.55, "natural_disaster": 0.30, "supply_disruption": 0.20},
                last_updated=datetime.utcnow()
            ),
            
            # Energy Suppliers
            SupplierProfile(
                supplier_id="aramco_saudi",
                supplier_name="Saudi Aramco",
                country="saudi_arabia", 
                region="middle_east",
                coordinates=(24.7136, 46.6753),
                capacity_rating=0.98,
                reliability_score=0.78,
                cost_index=0.85,
                quality_score=0.88,
                lead_time_days=14,
                certifications=["ISO9001", "API_Q1", "OHSAS18001"],
                specialties=["crude_oil", "natural_gas", "petrochemicals"],
                risk_factors={"geopolitical": 0.75, "natural_disaster": 0.15, "supply_disruption": 0.35},
                last_updated=datetime.utcnow()
            ),
            SupplierProfile(
                supplier_id="exxon_usa",
                supplier_name="ExxonMobil Corporation", 
                country="united_states",
                region="north_america",
                coordinates=(29.7604, -95.3698),
                capacity_rating=0.85,
                reliability_score=0.82,
                cost_index=1.12,
                quality_score=0.85,
                lead_time_days=21,
                certifications=["ISO9001", "API_Q1", "ISO14001"],
                specialties=["crude_oil", "natural_gas", "refined_products"],
                risk_factors={"geopolitical": 0.25, "natural_disaster": 0.45, "supply_disruption": 0.20},
                last_updated=datetime.utcnow()
            ),
            
            # Manufacturing Suppliers
            SupplierProfile(
                supplier_id="foxconn_china",
                supplier_name="Foxconn Technology Group",
                country="china",
                region="east_asia",
                coordinates=(22.3193, 114.1694), 
                capacity_rating=0.95,
                reliability_score=0.80,
                cost_index=0.75,
                quality_score=0.82,
                lead_time_days=45,
                certifications=["ISO9001", "ISO14001", "IATF16949"],
                specialties=["electronics_assembly", "consumer_goods", "automotive_parts"],
                risk_factors={"geopolitical": 0.55, "natural_disaster": 0.25, "supply_disruption": 0.30},
                last_updated=datetime.utcnow()
            ),
            SupplierProfile(
                supplier_id="siemens_germany",
                supplier_name="Siemens AG",
                country="germany",
                region="western_europe",
                coordinates=(48.1351, 11.5820),
                capacity_rating=0.85,
                reliability_score=0.90,
                cost_index=1.25,
                quality_score=0.95,
                lead_time_days=60,
                certifications=["ISO9001", "ISO14001", "IATF16949", "ISO27001"],
                specialties=["industrial_automation", "energy_systems", "medical_technology"],
                risk_factors={"geopolitical": 0.15, "natural_disaster": 0.10, "supply_disruption": 0.15},
                last_updated=datetime.utcnow()
            ),
            
            # Emerging Market Alternatives
            SupplierProfile(
                supplier_id="tata_india",
                supplier_name="Tata Group", 
                country="india",
                region="south_asia",
                coordinates=(19.0760, 72.8777),
                capacity_rating=0.70,
                reliability_score=0.75,
                cost_index=0.65,
                quality_score=0.78,
                lead_time_days=50,
                certifications=["ISO9001", "ISO14001", "OHSAS18001"],
                specialties=["steel", "automotive", "chemicals", "information_technology"],
                risk_factors={"geopolitical": 0.35, "natural_disaster": 0.30, "supply_disruption": 0.25},
                last_updated=datetime.utcnow()
            ),
            SupplierProfile(
                supplier_id="vale_brazil",
                supplier_name="Vale S.A.",
                country="brazil", 
                region="latin_america",
                coordinates=(-14.2350, -51.9253),
                capacity_rating=0.88,
                reliability_score=0.72,
                cost_index=0.80,
                quality_score=0.80,
                lead_time_days=35,
                certifications=["ISO9001", "ISO14001", "OHSAS18001"],
                specialties=["iron_ore", "nickel", "copper", "coal"],
                risk_factors={"geopolitical": 0.30, "natural_disaster": 0.25, "supply_disruption": 0.20},
                last_updated=datetime.utcnow()
            )
        ]
        
        return suppliers

    def _initialize_routes_database(self) -> List[TradeRoute]:
        """Initialize comprehensive trade routes database."""
        routes = [
            # Major Maritime Routes
            TradeRoute(
                route_id="asia_europe_suez",
                route_name="Asia-Europe via Suez Canal",
                origin_country="china",
                destination_country="germany",
                transit_countries=["singapore", "egypt", "netherlands"],
                transport_modes=["sea"],
                average_transit_days=35,
                cost_factor=1.0,  # Baseline
                reliability_score=0.85,
                risk_score=0.45,
                capacity_utilization=0.88,
                seasonal_factors={"Q1": 1.1, "Q2": 1.0, "Q3": 0.9, "Q4": 1.2}
            ),
            TradeRoute(
                route_id="asia_europe_cape",
                route_name="Asia-Europe via Cape of Good Hope",
                origin_country="china", 
                destination_country="germany",
                transit_countries=["singapore", "south_africa", "spain"],
                transport_modes=["sea"],
                average_transit_days=45,
                cost_factor=1.15,
                reliability_score=0.92,
                risk_score=0.25,
                capacity_utilization=0.65,
                seasonal_factors={"Q1": 0.9, "Q2": 1.0, "Q3": 1.1, "Q4": 0.95}
            ),
            
            # Trans-Pacific Routes
            TradeRoute(
                route_id="asia_us_pacific",
                route_name="Asia-US Pacific Route",
                origin_country="china",
                destination_country="united_states", 
                transit_countries=["japan"],
                transport_modes=["sea"],
                average_transit_days=14,
                cost_factor=0.90,
                reliability_score=0.88,
                risk_score=0.35,
                capacity_utilization=0.92,
                seasonal_factors={"Q1": 1.2, "Q2": 0.9, "Q3": 0.85, "Q4": 1.15}
            ),
            
            # Alternative Air Routes  
            TradeRoute(
                route_id="asia_europe_air",
                route_name="Asia-Europe Air Corridor",
                origin_country="china",
                destination_country="germany",
                transit_countries=["russia", "poland"],
                transport_modes=["air"],
                average_transit_days=2,
                cost_factor=4.5,
                reliability_score=0.95,
                risk_score=0.55,
                capacity_utilization=0.75,
                seasonal_factors={"Q1": 1.0, "Q2": 1.0, "Q3": 1.0, "Q4": 1.0}
            ),
            
            # Land Corridors
            TradeRoute(
                route_id="china_europe_belt_road",
                route_name="China-Europe Belt and Road",
                origin_country="china",
                destination_country="germany",
                transit_countries=["kazakhstan", "russia", "poland"],
                transport_modes=["rail"],
                average_transit_days=18,
                cost_factor=1.8,
                reliability_score=0.78,
                risk_score=0.65,
                capacity_utilization=0.60,
                seasonal_factors={"Q1": 0.85, "Q2": 1.1, "Q3": 1.15, "Q4": 0.90}
            ),
            
            # Regional Routes
            TradeRoute(
                route_id="nafta_corridor",
                route_name="NAFTA Trade Corridor",
                origin_country="mexico",
                destination_country="united_states",
                transit_countries=[],
                transport_modes=["road", "rail"],
                average_transit_days=3,
                cost_factor=0.70,
                reliability_score=0.90,
                risk_score=0.20,
                capacity_utilization=0.85,
                seasonal_factors={"Q1": 1.0, "Q2": 1.0, "Q3": 1.0, "Q4": 1.0}
            ),
            
            # Energy Routes
            TradeRoute(
                route_id="middle_east_asia_lng",
                route_name="Middle East-Asia LNG Route",
                origin_country="qatar",
                destination_country="japan",
                transit_countries=["united_arab_emirates", "india"],
                transport_modes=["sea"],
                average_transit_days=21,
                cost_factor=1.1,
                reliability_score=0.80,
                risk_score=0.50,
                capacity_utilization=0.90,
                seasonal_factors={"Q1": 1.3, "Q2": 0.8, "Q3": 0.7, "Q4": 1.2}
            )
        ]
        
        return routes

    def generate_sourcing_recommendations(
        self,
        disrupted_suppliers: List[str] = None,
        affected_commodities: List[str] = None, 
        target_regions: List[str] = None,
        budget_constraint_percent: float = 1.2,  # Max 20% cost increase
        lead_time_constraint_days: int = 90,
        quality_threshold: float = 0.75,
        max_recommendations: int = 10
    ) -> List[SourcingRecommendation]:
        """Generate comprehensive sourcing recommendations based on constraints."""
        
        recommendations = []
        current_time = datetime.utcnow()
        
        # Set defaults if not provided
        disrupted_suppliers = disrupted_suppliers or []
        affected_commodities = affected_commodities or ["semiconductors", "crude_oil", "manufactured_goods"]
        target_regions = target_regions or ["east_asia", "north_america", "western_europe"]
        
        # 1. Supplier Diversification Recommendations
        supplier_recs = self._generate_supplier_diversification(
            disrupted_suppliers,
            affected_commodities,
            target_regions,
            budget_constraint_percent,
            lead_time_constraint_days,
            quality_threshold
        )
        recommendations.extend(supplier_recs)
        
        # 2. Route Optimization Recommendations
        route_recs = self._generate_route_optimization(
            affected_commodities,
            target_regions,
            budget_constraint_percent
        )
        recommendations.extend(route_recs)
        
        # 3. Inventory Buffering Recommendations
        inventory_recs = self._generate_inventory_buffering(
            affected_commodities,
            disrupted_suppliers
        )
        recommendations.extend(inventory_recs)
        
        # 4. Nearshoring/Reshoring Recommendations 
        nearshoring_recs = self._generate_nearshoring_recommendations(
            affected_commodities,
            target_regions
        )
        recommendations.extend(nearshoring_recs)
        
        # 5. Contract and Hedging Recommendations
        contract_recs = self._generate_contract_hedging(
            affected_commodities,
            disrupted_suppliers
        )
        recommendations.extend(contract_recs)
        
        # Sort by priority and confidence
        recommendations.sort(key=lambda x: (
            {"critical": 4, "high": 3, "medium": 2, "low": 1}[x.priority.value],
            x.confidence_score
        ), reverse=True)
        
        return recommendations[:max_recommendations]

    def _generate_supplier_diversification(
        self,
        disrupted_suppliers: List[str],
        commodities: List[str], 
        regions: List[str],
        budget_constraint: float,
        lead_time_constraint: int,
        quality_threshold: float
    ) -> List[SourcingRecommendation]:
        """Generate supplier diversification recommendations."""
        
        recommendations = []
        
        for commodity in commodities:
            strategy = self.commodity_strategies.get(commodity, {})
            
            # Find alternative suppliers for this commodity
            alternative_suppliers = []
            for supplier in self.suppliers_db:
                if (commodity in supplier.specialties and
                    supplier.supplier_id not in disrupted_suppliers and
                    supplier.cost_index <= budget_constraint and
                    supplier.lead_time_days <= lead_time_constraint and
                    supplier.quality_score >= quality_threshold):
                    alternative_suppliers.append(supplier)
            
            if not alternative_suppliers:
                continue
                
            # Sort by composite score
            alternative_suppliers.sort(
                key=lambda s: (
                    s.reliability_score * 0.3 +
                    s.quality_score * 0.3 +
                    (1.0 / s.cost_index) * 0.2 +  # Lower cost is better
                    (1.0 - max(s.risk_factors.values())) * 0.2  # Lower risk is better
                ), reverse=True
            )
            
            # Calculate diversification priority
            critical_suppliers = strategy.get("critical_suppliers", [])
            disrupted_critical = [s for s in disrupted_suppliers if any(region in s for region in critical_suppliers)]
            
            if disrupted_critical:
                priority = SourcingPriority.CRITICAL
            elif len(disrupted_suppliers) >= 2:
                priority = SourcingPriority.HIGH
            else:
                priority = SourcingPriority.MEDIUM
            
            # Calculate cost impact
            baseline_cost = 1.0  # Baseline supplier cost index
            new_cost = sum(s.cost_index for s in alternative_suppliers[:3]) / min(3, len(alternative_suppliers))
            cost_impact = ((new_cost - baseline_cost) / baseline_cost) * 100
            
            # Calculate risk reduction
            baseline_risk = 0.5  # Assume moderate baseline risk
            new_risk = sum(max(s.risk_factors.values()) for s in alternative_suppliers[:3]) / min(3, len(alternative_suppliers))
            risk_reduction = ((baseline_risk - new_risk) / baseline_risk) * 100
            
            recommendation = SourcingRecommendation(
                recommendation_id=f"supplier_div_{commodity}_{int(datetime.utcnow().timestamp())}",
                recommendation_type=RecommendationType.SUPPLIER_DIVERSIFICATION,
                priority=priority,
                title=f"Diversify {commodity.replace('_', ' ').title()} Suppliers",
                description=f"Reduce dependency on {len(disrupted_suppliers)} disrupted suppliers by engaging {len(alternative_suppliers)} alternative suppliers with strong reliability and quality scores.",
                affected_commodities=[commodity],
                target_regions=list(set([s.region for s in alternative_suppliers[:5]])),
                implementation_timeline_weeks=8,
                cost_impact_percent=cost_impact,
                risk_reduction_percent=max(0, risk_reduction),
                confidence_score=0.85 if len(alternative_suppliers) >= 3 else 0.65,
                alternative_suppliers=alternative_suppliers[:5],
                alternative_routes=[],
                implementation_steps=[
                    f"Evaluate and qualify top {min(3, len(alternative_suppliers))} alternative suppliers",
                    "Initiate supplier risk assessments and due diligence",
                    "Negotiate trial contracts with favorable terms",
                    "Establish secondary supplier relationships",
                    "Implement dual-sourcing strategy for critical components"
                ],
                success_metrics=[
                    "Reduce single-supplier dependency to <50%",
                    "Maintain quality standards above 90%",
                    f"Keep cost increase within {budget_constraint * 100:.0f}% of baseline",
                    "Achieve 95% supply availability"
                ],
                dependencies=[
                    "Legal approval for new supplier contracts",
                    "Quality assurance team capacity",
                    "Integration with existing supply chain systems"
                ],
                created_at=datetime.utcnow()
            )
            
            recommendations.append(recommendation)
        
        return recommendations

    def _generate_route_optimization(
        self,
        commodities: List[str],
        regions: List[str], 
        budget_constraint: float
    ) -> List[SourcingRecommendation]:
        """Generate trade route optimization recommendations."""
        
        recommendations = []
        
        # Find alternative routes within budget
        viable_routes = [
            route for route in self.routes_db
            if route.cost_factor <= budget_constraint
        ]
        
        if not viable_routes:
            return recommendations
        
        # Group routes by origin-destination pairs
        route_groups = {}
        for route in viable_routes:
            key = f"{route.origin_country}_{route.destination_country}"
            if key not in route_groups:
                route_groups[key] = []
            route_groups[key].append(route)
        
        for route_pair, routes in route_groups.items():
            if len(routes) < 2:  # Need alternatives
                continue
            
            # Sort by composite score (reliability vs cost vs risk)
            routes.sort(key=lambda r: (
                r.reliability_score * 0.4 +
                (1.0 / r.cost_factor) * 0.3 +  # Lower cost is better  
                (1.0 - r.risk_score) * 0.3    # Lower risk is better
            ), reverse=True)
            
            primary_route = routes[0]
            
            # Calculate potential impact
            baseline_cost = 1.0
            cost_impact = ((primary_route.cost_factor - baseline_cost) / baseline_cost) * 100
            
            baseline_risk = 0.5  
            risk_reduction = ((baseline_risk - primary_route.risk_score) / baseline_risk) * 100
            
            recommendation = SourcingRecommendation(
                recommendation_id=f"route_opt_{route_pair}_{int(datetime.utcnow().timestamp())}",
                recommendation_type=RecommendationType.ROUTE_OPTIMIZATION,
                priority=SourcingPriority.HIGH if primary_route.risk_score < 0.3 else SourcingPriority.MEDIUM,
                title=f"Optimize {route_pair.replace('_', '-').title()} Trade Route",
                description=f"Shift to more reliable {'/'.join(primary_route.transport_modes)} route with {primary_route.average_transit_days} day transit time and {(primary_route.reliability_score * 100):.0f}% reliability.",
                affected_commodities=commodities,
                target_regions=[primary_route.origin_country, primary_route.destination_country],
                implementation_timeline_weeks=4,
                cost_impact_percent=cost_impact,
                risk_reduction_percent=max(0, risk_reduction),
                confidence_score=0.80,
                alternative_suppliers=[],
                alternative_routes=routes[:3],
                implementation_steps=[
                    f"Negotiate shipping contracts with {'/'.join(primary_route.transport_modes)} carriers",
                    "Update logistics planning systems with new route parameters", 
                    "Conduct trial shipments to validate transit times",
                    "Implement route monitoring and tracking systems",
                    "Establish backup route protocols"
                ],
                success_metrics=[
                    f"Achieve {(primary_route.reliability_score * 100):.0f}% on-time delivery",
                    f"Reduce transit time variability to <{primary_route.average_transit_days * 0.15:.0f} days",
                    "Maintain route cost within budget constraints",
                    "Zero critical shipment delays"
                ],
                dependencies=[
                    "Carrier capacity availability",
                    "Customs and regulatory approvals",
                    "Insurance coverage for new routes"
                ],
                created_at=datetime.utcnow()
            )
            
            recommendations.append(recommendation)
        
        return recommendations

    def _generate_inventory_buffering(
        self,
        commodities: List[str],
        disrupted_suppliers: List[str]
    ) -> List[SourcingRecommendation]:
        """Generate strategic inventory buffering recommendations."""
        
        recommendations = []
        
        for commodity in commodities:
            strategy = self.commodity_strategies.get(commodity, {})
            lead_time_sensitivity = strategy.get("lead_time_sensitivity", 0.7)
            
            # Calculate recommended buffer levels
            base_buffer_weeks = 4
            disruption_multiplier = 1 + (len(disrupted_suppliers) * 0.5)
            sensitivity_multiplier = 1 + lead_time_sensitivity
            
            recommended_buffer = base_buffer_weeks * disruption_multiplier * sensitivity_multiplier
            
            # Calculate costs and benefits
            inventory_cost_increase = recommended_buffer * 0.15  # 15% cost per week of buffer
            risk_reduction = min(85, recommended_buffer * 8)  # Diminishing returns
            
            priority = SourcingPriority.CRITICAL if lead_time_sensitivity > 0.8 else SourcingPriority.HIGH
            
            recommendation = SourcingRecommendation(
                recommendation_id=f"inventory_buffer_{commodity}_{int(datetime.utcnow().timestamp())}",
                recommendation_type=RecommendationType.INVENTORY_BUFFERING,
                priority=priority,
                title=f"Build Strategic {commodity.replace('_', ' ').title()} Inventory Buffer",
                description=f"Increase {commodity} inventory buffer to {recommended_buffer:.1f} weeks to mitigate supply disruption risks and reduce lead time sensitivity.",
                affected_commodities=[commodity],
                target_regions=["global"],
                implementation_timeline_weeks=6,
                cost_impact_percent=inventory_cost_increase,
                risk_reduction_percent=risk_reduction,
                confidence_score=0.75,
                alternative_suppliers=[],
                alternative_routes=[],
                implementation_steps=[
                    f"Analyze current {commodity} consumption patterns and variability",
                    f"Identify optimal buffer stock levels for {commodity}",
                    "Secure additional warehouse/storage capacity",
                    "Implement inventory management system updates",
                    "Establish reorder triggers and safety stock policies"
                ],
                success_metrics=[
                    f"Maintain {recommended_buffer:.1f} weeks of {commodity} inventory",
                    "Zero stockouts during supply disruptions",
                    "Reduce lead time variance to <20%",
                    "Optimize inventory turnover within cost constraints"
                ],
                dependencies=[
                    "Warehouse capacity expansion",
                    "Working capital availability",
                    "Inventory management system updates"
                ],
                created_at=datetime.utcnow()
            )
            
            recommendations.append(recommendation)
        
        return recommendations

    def _generate_nearshoring_recommendations(
        self,
        commodities: List[str],
        target_regions: List[str]
    ) -> List[SourcingRecommendation]:
        """Generate nearshoring/reshoring recommendations."""
        
        recommendations = []
        
        # Identify nearshoring opportunities for each region
        nearshoring_options = {
            "north_america": ["mexico", "canada", "united_states"],
            "western_europe": ["poland", "czech_republic", "romania", "spain"],
            "east_asia": ["vietnam", "thailand", "philippines", "malaysia"]
        }
        
        for region in target_regions:
            if region not in nearshoring_options:
                continue
                
            nearshore_countries = nearshoring_options[region]
            
            # Find suppliers in nearshore countries
            nearshore_suppliers = [
                supplier for supplier in self.suppliers_db
                if supplier.country in nearshore_countries
            ]
            
            if not nearshore_suppliers:
                continue
            
            # Calculate nearshoring benefits
            cost_increase = 15  # Typical 15% cost increase for nearshoring
            risk_reduction = 35  # Significant risk reduction from proximity
            lead_time_reduction = 40  # Major lead time improvement
            
            recommendation = SourcingRecommendation(
                recommendation_id=f"nearshoring_{region}_{int(datetime.utcnow().timestamp())}",
                recommendation_type=RecommendationType.NEARSHORING,
                priority=SourcingPriority.HIGH,
                title=f"Nearshoring Strategy for {region.replace('_', ' ').title()}",
                description=f"Shift production closer to {region} using suppliers in {', '.join(nearshore_countries)} to reduce lead times and geopolitical risks.",
                affected_commodities=commodities,
                target_regions=[region] + nearshore_countries,
                implementation_timeline_weeks=24,  # Longer implementation
                cost_impact_percent=cost_increase,
                risk_reduction_percent=risk_reduction,
                confidence_score=0.70,
                alternative_suppliers=nearshore_suppliers[:5],
                alternative_routes=[],
                implementation_steps=[
                    f"Conduct feasibility study for {region} nearshoring",
                    "Identify and qualify nearshore supplier partners",
                    "Negotiate long-term supply agreements",
                    "Establish local quality assurance capabilities",
                    "Gradually transition production volumes"
                ],
                success_metrics=[
                    f"Reduce average lead times by {lead_time_reduction}%",
                    f"Achieve {risk_reduction}% reduction in geopolitical risk exposure",
                    "Maintain quality standards above 90%",
                    "Establish resilient regional supply base"
                ],
                dependencies=[
                    "Regional supplier capacity development",
                    "Local regulatory compliance",
                    "Technology transfer and training",
                    "Significant capital investment commitment"
                ],
                created_at=datetime.utcnow()
            )
            
            recommendations.append(recommendation)
        
        return recommendations

    def _generate_contract_hedging(
        self,
        commodities: List[str],
        disrupted_suppliers: List[str]
    ) -> List[SourcingRecommendation]:
        """Generate contract and hedging strategy recommendations."""
        
        recommendations = []
        
        # High-volatility commodities benefit most from hedging
        volatile_commodities = ["crude_oil", "natural_gas", "copper", "steel", "rare_earth_metals"]
        
        hedging_commodities = [c for c in commodities if c in volatile_commodities]
        
        if not hedging_commodities:
            return recommendations
        
        for commodity in hedging_commodities:
            # Calculate hedging benefits based on volatility
            volatility_factor = {"crude_oil": 0.35, "natural_gas": 0.45, "copper": 0.25}.get(commodity, 0.30)
            
            cost_impact = 3.5  # Typical hedging cost 3-5% of value
            risk_reduction = volatility_factor * 60  # Reduce price volatility risk
            
            recommendation = SourcingRecommendation(
                recommendation_id=f"hedging_{commodity}_{int(datetime.utcnow().timestamp())}",
                recommendation_type=RecommendationType.CONTRACT_HEDGING,
                priority=SourcingPriority.HIGH if volatility_factor > 0.3 else SourcingPriority.MEDIUM,
                title=f"Implement {commodity.replace('_', ' ').title()} Price Hedging Strategy",
                description=f"Use financial instruments to hedge against {commodity} price volatility and secure predictable supply costs through contract structures.",
                affected_commodities=[commodity],
                target_regions=["global"],
                implementation_timeline_weeks=12,
                cost_impact_percent=cost_impact,
                risk_reduction_percent=risk_reduction,
                confidence_score=0.80,
                alternative_suppliers=[],
                alternative_routes=[],
                implementation_steps=[
                    f"Analyze {commodity} price volatility and exposure",
                    "Develop hedging strategy and risk management framework",
                    "Engage financial institutions for hedging instruments",
                    "Establish long-term supply contracts with price corridors",
                    "Implement regular hedging performance monitoring"
                ],
                success_metrics=[
                    f"Reduce {commodity} price volatility exposure by {risk_reduction:.0f}%",
                    "Maintain hedging costs below 5% of commodity value",
                    "Achieve 95% price predictability for planning cycles",
                    "Optimize contract renewal timing for cost efficiency"
                ],
                dependencies=[
                    "Financial risk management capabilities",
                    "Regulatory compliance for financial instruments",
                    "Supplier willingness for structured contracts",
                    "Treasury and finance team capacity"
                ],
                created_at=datetime.utcnow()
            )
            
            recommendations.append(recommendation)
        
        return recommendations

    def evaluate_supplier_performance(
        self,
        supplier_id: str,
        evaluation_period_days: int = 90
    ) -> Dict[str, Any]:
        """Evaluate supplier performance for recommendation updates."""
        
        supplier = next((s for s in self.suppliers_db if s.supplier_id == supplier_id), None)
        if not supplier:
            return {"error": "Supplier not found"}
        
        # Simulate performance evaluation
        current_performance = {
            "delivery_performance": {
                "on_time_delivery_rate": 0.88,
                "early_delivery_rate": 0.12,
                "late_delivery_rate": 0.05,
                "average_delay_days": 1.2
            },
            "quality_performance": {
                "quality_score": supplier.quality_score,
                "defect_rate": 0.02,
                "return_rate": 0.01,
                "customer_satisfaction": 0.85
            },
            "cost_performance": {
                "cost_index": supplier.cost_index,
                "price_stability": 0.78,
                "cost_reduction_initiatives": 3,
                "total_cost_of_ownership": supplier.cost_index * 1.15
            },
            "risk_performance": {
                "risk_incidents": 1,
                "business_continuity_score": 0.82,
                "compliance_score": 0.90,
                "financial_stability": 0.85
            }
        }
        
        # Calculate overall performance score
        scores = [
            current_performance["delivery_performance"]["on_time_delivery_rate"],
            current_performance["quality_performance"]["quality_score"],
            1.0 / current_performance["cost_performance"]["cost_index"],  # Lower cost is better
            current_performance["risk_performance"]["business_continuity_score"]
        ]
        
        overall_score = sum(scores) / len(scores)
        
        # Performance trend (simulated)
        trend_direction = "improving" if overall_score > 0.80 else "stable" if overall_score > 0.70 else "declining"
        
        return {
            "supplier_id": supplier_id,
            "supplier_name": supplier.supplier_name,
            "evaluation_period_days": evaluation_period_days,
            "overall_performance_score": round(overall_score, 3),
            "performance_trend": trend_direction,
            "detailed_performance": current_performance,
            "recommendations": [
                "Continue strategic partnership" if overall_score > 0.85 else
                "Monitor performance closely" if overall_score > 0.70 else
                "Consider alternative suppliers"
            ],
            "evaluated_at": datetime.utcnow().isoformat() + "Z"
        }

    def get_sourcing_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive sourcing optimization summary."""
        
        # Calculate regional supplier distribution
        regional_distribution = {}
        for supplier in self.suppliers_db:
            region = supplier.region
            if region not in regional_distribution:
                regional_distribution[region] = {"count": 0, "avg_reliability": 0, "avg_cost_index": 0}
            
            regional_distribution[region]["count"] += 1
            regional_distribution[region]["avg_reliability"] += supplier.reliability_score
            regional_distribution[region]["avg_cost_index"] += supplier.cost_index
        
        # Calculate averages
        for region, data in regional_distribution.items():
            count = data["count"]
            data["avg_reliability"] = round(data["avg_reliability"] / count, 3)
            data["avg_cost_index"] = round(data["avg_cost_index"] / count, 3)
        
        # Route analysis
        route_analysis = {
            "total_routes": len(self.routes_db),
            "avg_transit_days": round(sum(r.average_transit_days for r in self.routes_db) / len(self.routes_db), 1),
            "avg_reliability_score": round(sum(r.reliability_score for r in self.routes_db) / len(self.routes_db), 3),
            "transport_mode_distribution": {}
        }
        
        # Transport mode distribution
        mode_counts = {}
        for route in self.routes_db:
            for mode in route.transport_modes:
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
        route_analysis["transport_mode_distribution"] = mode_counts
        
        # Risk assessment
        risk_assessment = {
            "high_risk_regions": [region for region, risk in self.regional_risks.items() if risk > 0.5],
            "avg_regional_risk": round(sum(self.regional_risks.values()) / len(self.regional_risks), 3),
            "commodity_risk_levels": {
                commodity: data.get("diversification_priority", "medium")
                for commodity, data in self.commodity_strategies.items()
            }
        }
        
        return {
            "summary_generated_at": datetime.utcnow().isoformat() + "Z",
            "supplier_analysis": {
                "total_suppliers": len(self.suppliers_db),
                "regional_distribution": regional_distribution,
                "avg_reliability_score": round(sum(s.reliability_score for s in self.suppliers_db) / len(self.suppliers_db), 3),
                "avg_cost_index": round(sum(s.cost_index for s in self.suppliers_db) / len(self.suppliers_db), 3)
            },
            "route_analysis": route_analysis,
            "risk_assessment": risk_assessment,
            "optimization_opportunities": [
                "Diversify supplier base in high-risk regions",
                "Develop alternative routes for critical corridors", 
                "Implement strategic inventory buffers for volatile commodities",
                "Consider nearshoring for risk reduction",
                "Establish hedging strategies for price-volatile inputs"
            ]
        }


# Singleton instance
_sourcing_engine = None


def get_sourcing_engine() -> SourcingRecommendationEngine:
    """Get the sourcing recommendation engine instance."""
    global _sourcing_engine
    if _sourcing_engine is None:
        _sourcing_engine = SourcingRecommendationEngine()
    return _sourcing_engine