#!/usr/bin/env python3
"""
Generate Synthetic Training Data for ML Models

Creates realistic synthetic training data that matches the distribution
of real supply chain, economic, and geopolitical data for ML model training.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import random
from typing import List, Dict, Any

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import SessionLocal, engine
from app.models import (
    CascadeEvent, SupplyChainNode, SupplyChainRelationship,
    SectorVulnerabilityAssessment, ACLEDEvent, ResilienceMetric
)
from sqlalchemy.orm import Session
from sqlalchemy import text

def generate_supply_chain_nodes(session: Session, count: int = 500) -> List[Dict]:
    """Generate synthetic supply chain nodes with realistic characteristics."""
    
    # Real-world supply chain hubs and characteristics
    major_hubs = [
        ("Shanghai", "CHN", "port", 31.2304, 121.4737),
        ("Singapore", "SGP", "port", 1.3521, 103.8198),
        ("Rotterdam", "NLD", "port", 51.9244, 4.4777),
        ("Los Angeles", "USA", "port", 34.0522, -118.2437),
        ("Hamburg", "DEU", "port", 53.5511, 9.9937),
        ("Dubai", "ARE", "port", 25.2048, 55.2708),
        ("Hong Kong", "HKG", "port", 22.3193, 114.1694),
        ("Antwerp", "BEL", "port", 51.2194, 4.4025),
        ("Long Beach", "USA", "port", 33.7701, -118.1937),
        ("Busan", "KOR", "port", 35.1796, 129.0756),
        ("Memphis", "USA", "airport", 35.1495, -90.0490),  # FedEx hub
        ("Louisville", "USA", "airport", 38.2527, -85.7585),  # UPS hub
        ("Frankfurt", "DEU", "airport", 50.1109, 8.6821),
        ("Tokyo", "JPN", "port", 35.6762, 139.6503),
        ("Chennai", "IND", "port", 13.0827, 80.2707)
    ]
    
    # Industries and their typical characteristics
    industries = {
        "electronics": {"risk_multiplier": 1.2, "typical_value": 50000},
        "automotive": {"risk_multiplier": 1.1, "typical_value": 30000},
        "textiles": {"risk_multiplier": 0.8, "typical_value": 5000},
        "chemicals": {"risk_multiplier": 1.4, "typical_value": 25000},
        "pharmaceuticals": {"risk_multiplier": 1.6, "typical_value": 100000},
        "food": {"risk_multiplier": 0.9, "typical_value": 8000},
        "energy": {"risk_multiplier": 1.3, "typical_value": 75000},
        "machinery": {"risk_multiplier": 1.0, "typical_value": 40000},
        "raw_materials": {"risk_multiplier": 0.7, "typical_value": 3000},
        "consumer_goods": {"risk_multiplier": 0.9, "typical_value": 15000}
    }
    
    nodes = []
    
    # Create major hubs
    for i, (name, country, node_type, lat, lng) in enumerate(major_hubs):
        industry = random.choice(list(industries.keys()))
        industry_info = industries[industry]
        
        node = SupplyChainNode(
            identifier=f"hub_{i:03d}",
            name=name,
            node_type=node_type,
            country=country,
            latitude=lat,
            longitude=lng,
            industry_sector=industry,
            overall_risk_score=random.uniform(0.1, 0.8) * industry_info["risk_multiplier"],
            financial_health_score=random.uniform(0.7, 1.0),
            operational_risk_score=random.uniform(0.1, 0.6),
            geopolitical_risk_score=random.uniform(0.1, 0.5),
            tier_level=1,  # Major hubs are tier 1
            created_at=datetime.utcnow()
        )
        nodes.append(node)
    
    # Generate additional nodes
    countries = ["USA", "CHN", "DEU", "JPN", "GBR", "FRA", "IND", "BRA", "KOR", "ITA", 
                "CAN", "MEX", "THA", "VNM", "MYS", "IDN", "TUR", "POL", "ESP", "NLD"]
    
    for i in range(len(major_hubs), count):
        country = random.choice(countries)
        industry = random.choice(list(industries.keys()))
        industry_info = industries[industry]
        node_type = random.choice(["factory", "warehouse", "distribution_center", "supplier"])
        
        # Generate realistic coordinates based on country
        if country == "USA":
            lat = random.uniform(25.0, 49.0)
            lng = random.uniform(-125.0, -66.0)
        elif country == "CHN":
            lat = random.uniform(18.0, 54.0)
            lng = random.uniform(73.0, 135.0)
        elif country == "DEU":
            lat = random.uniform(47.0, 55.0)
            lng = random.uniform(5.0, 15.0)
        else:
            lat = random.uniform(-60.0, 70.0)
            lng = random.uniform(-180.0, 180.0)
        
        node = SupplyChainNode(
            identifier=f"node_{i:03d}",
            name=f"{node_type.title()} {i}",
            node_type=node_type,
            country=country,
            latitude=lat,
            longitude=lng,
            industry_sector=industry,
            overall_risk_score=random.uniform(0.05, 1.0) * industry_info["risk_multiplier"],
            financial_health_score=random.uniform(0.3, 0.9),
            operational_risk_score=random.uniform(0.1, 0.8),
            geopolitical_risk_score=random.uniform(0.05, 0.7),
            tier_level=random.randint(1, 4),
            created_at=datetime.utcnow()
        )
        nodes.append(node)
    
    # Add to database
    session.add_all(nodes)
    session.commit()
    print(f"Generated {len(nodes)} supply chain nodes")
    return nodes

def generate_cascade_events(session: Session, nodes: List[SupplyChainNode], count: int = 1000) -> List[CascadeEvent]:
    """Generate realistic cascade events based on real disruption patterns."""
    
    # Real-world disruption types and their characteristics
    disruption_types = {
        "port_closure": {
            "severity_range": (0.6, 1.0),
            "duration_days": (3, 21),
            "affected_radius": 200,
            "probability": 0.1
        },
        "natural_disaster": {
            "severity_range": (0.7, 1.0),
            "duration_days": (7, 60),
            "affected_radius": 500,
            "probability": 0.15
        },
        "labor_strike": {
            "severity_range": (0.4, 0.8),
            "duration_days": (1, 14),
            "affected_radius": 50,
            "probability": 0.2
        },
        "cyber_attack": {
            "severity_range": (0.3, 0.9),
            "duration_days": (1, 7),
            "affected_radius": 0,  # Can affect connected nodes regardless of distance
            "probability": 0.1
        },
        "supply_shortage": {
            "severity_range": (0.2, 0.7),
            "duration_days": (7, 45),
            "affected_radius": 100,
            "probability": 0.25
        },
        "transport_disruption": {
            "severity_range": (0.3, 0.8),
            "duration_days": (1, 10),
            "affected_radius": 300,
            "probability": 0.2
        }
    }
    
    events = []
    start_date = datetime.utcnow() - timedelta(days=365)
    
    for i in range(count):
        # Select random disruption type
        disruption_type = np.random.choice(list(disruption_types.keys()), 
                                         p=[disruption_types[dt]["probability"] for dt in disruption_types.keys()])
        disruption_info = disruption_types[disruption_type]
        
        # Select affected node
        source_node = random.choice(nodes)
        
        # Generate event characteristics
        severity = random.uniform(*disruption_info["severity_range"])
        duration = random.randint(*disruption_info["duration_days"])
        event_date = start_date + timedelta(days=random.randint(0, 365))
        
        # Calculate cascade effects based on node importance and disruption severity
        cascade_size = int(severity * (1.0 - source_node.overall_risk_score) * random.uniform(1, 10))
        recovery_time = duration + random.randint(0, duration // 2)
        
        event = CascadeEvent(
            cascade_id=f"cascade_{i:04d}",
            event_type="disruption",
            severity=["low", "medium", "high", "critical"][int(severity * 4) % 4],
            title=f"{disruption_type.replace('_', ' ').title()} at {source_node.name}",
            description=f"Supply chain disruption caused by {disruption_type} affecting {source_node.name}",
            trigger_event=disruption_type,
            affected_countries=[source_node.country] if source_node.country else [],
            affected_sectors=[source_node.industry_sector] if source_node.industry_sector else [],
            affected_nodes=[source_node.id],
            estimated_cost_usd=severity * 1000000 * random.uniform(0.1, 5.0),
            affected_companies_count=cascade_size,
            supply_disruption_percentage=severity * 100 * random.uniform(0.05, 0.4),
            recovery_time_days=recovery_time,
            event_start=event_date,
            event_end=event_date + timedelta(days=duration),
            propagation_speed=random.choice(["immediate", "hours", "days", "weeks"]),
            cascade_depth=random.randint(1, 5),
            confidence_level=random.uniform(0.5, 1.0) * 100,
            detection_method=random.choice(["manual", "automated", "ml_prediction"]),
            status=random.choice(["active", "resolved", "ongoing"])
        )
        events.append(event)
    
    session.add_all(events)
    session.commit()
    print(f"Generated {len(events)} cascade events")
    return events

def generate_acled_events(session: Session, count: int = 2000) -> List[ACLEDEvent]:
    """Generate synthetic ACLED-style geopolitical events."""
    
    # Real conflict hotspots and their characteristics
    conflict_regions = [
        ("Middle East", 25.0, 45.0, 35.0, 55.0, 0.8),  # lat_min, lng_min, lat_max, lng_max, intensity
        ("Eastern Europe", 45.0, 20.0, 55.0, 40.0, 0.6),
        ("Sub-Saharan Africa", -20.0, -10.0, 15.0, 50.0, 0.7),
        ("South Asia", 5.0, 60.0, 35.0, 95.0, 0.5),
        ("Southeast Asia", -10.0, 90.0, 25.0, 140.0, 0.4),
        ("Latin America", -30.0, -85.0, 15.0, -35.0, 0.3)
    ]
    
    event_types = [
        "Riots", "Protests", "Violence against civilians", "Battles",
        "Explosions/Remote violence", "Strategic developments"
    ]
    
    events = []
    start_date = datetime.utcnow() - timedelta(days=365)
    
    for i in range(count):
        # Select region based on conflict intensity
        region_weights = [region[5] for region in conflict_regions]
        region_idx = np.random.choice(len(conflict_regions), p=np.array(region_weights)/sum(region_weights))
        region = conflict_regions[region_idx]
        region_name, lat_min, lng_min, lat_max, lng_max, intensity = region
        
        # Generate location within region
        latitude = random.uniform(lat_min, lat_max)
        longitude = random.uniform(lng_min, lng_max)
        
        # Generate event characteristics
        event_type = random.choice(event_types)
        fatalities = max(0, int(np.random.poisson(intensity * 5)))
        event_date = start_date + timedelta(days=random.randint(0, 365))
        
        event = ACLEDEvent(
            acled_event_id=f"acled_{i:04d}",
            event_date=event_date,
            event_type=event_type,
            sub_event_type=random.choice(["Armed clash", "Remote violence", "Protest", "Riots", "Strategic development"]),
            country=f"Country_{random.randint(1, 50)}",
            region=region_name,
            location=f"Location in {region_name}",
            latitude=latitude,
            longitude=longitude,
            fatalities=fatalities,
            severity_score=fatalities / 100.0 * random.uniform(0.5, 2.0),
            supply_chain_relevance=random.choice([True, False]),
            economic_impact_estimate=fatalities * 10000 * random.uniform(0.1, 5.0),
            actors_involved=[f"Actor_{random.randint(1, 10)}"] if random.random() > 0.3 else [],
            sectors_affected=[random.choice(["transportation", "energy", "manufacturing", "agriculture"])] if random.random() > 0.5 else [],
            transportation_impact=random.choice([True, False]),
            port_impact=random.choice([True, False])
        )
        events.append(event)
    
    session.add_all(events)
    session.commit()
    print(f"Generated {len(events)} ACLED events")
    return events

def generate_vulnerability_assessments(session: Session, count: int = 50) -> List[SectorVulnerabilityAssessment]:
    """Generate sector vulnerability assessments."""
    
    sectors = [
        "electronics", "automotive", "pharmaceuticals", "food_agriculture",
        "energy", "chemicals", "textiles", "machinery", "aerospace", "healthcare"
    ]
    
    assessments = []
    
    for sector in sectors:
        for i in range(count // len(sectors)):
            assessment = SectorVulnerabilityAssessment(
                assessment_id=f"vuln_{sector}_{i:02d}",
                sector=sector,
                sector_name=f"{sector.replace('_', ' ').title()} Sector",
                assessment_date=datetime.utcnow() - timedelta(days=random.randint(0, 90)),
                assessment_version="1.0",
                assessment_methodology="Synthetic Risk Assessment",
                overall_risk_score=random.uniform(0.1, 0.9),
                vulnerability_count=random.randint(5, 50),
                critical_vulnerabilities=random.randint(0, 5),
                high_vulnerabilities=random.randint(1, 10),
                medium_vulnerabilities=random.randint(2, 15),
                low_vulnerabilities=random.randint(5, 25),
                complexity_score=random.uniform(0.1, 1.0),
                globalization_index=random.uniform(0.1, 1.0),
                regulatory_burden=random.uniform(0.1, 1.0),
                technology_dependency=random.uniform(0.1, 1.0),
                environmental_sensitivity=random.uniform(0.1, 1.0),
                geopolitical_exposure=random.uniform(0.1, 1.0)
            )
            assessments.append(assessment)
    
    session.add_all(assessments)
    session.commit()
    print(f"Generated {len(assessments)} vulnerability assessments")
    return assessments

def generate_supply_chain_relationships(session: Session, nodes: List[SupplyChainNode]) -> List[SupplyChainRelationship]:
    """Generate realistic supply chain relationships between nodes."""
    
    relationships = []
    
    # Create hub-to-node relationships (major hubs connect to many nodes)
    hub_nodes = [node for node in nodes if node.node_type in ["port", "airport"] and node.overall_risk_score < 0.3]
    
    for hub in hub_nodes[:15]:  # Top 15 hubs
        # Each hub connects to 20-50 other nodes
        num_connections = random.randint(20, 50)
        connected_nodes = random.sample([n for n in nodes if n != hub], num_connections)
        
        for connected_node in connected_nodes:
            # Calculate relationship strength based on distance and importance
            distance = np.sqrt((hub.latitude - connected_node.latitude)**2 + 
                             (hub.longitude - connected_node.longitude)**2)
            strength = max(0.1, 1.0 - (distance / 100.0)) * random.uniform(0.5, 1.5)
            
            relationship = SupplyChainRelationship(
                upstream_node_id=hub.id,
                downstream_node_id=connected_node.id,
                relationship_type="supplier",
                relationship_strength=min(1.0, strength),
                trade_volume_usd=random.randint(10000, 10000000),
                percentage_of_downstream_supply=random.uniform(0.05, 0.3),
                criticality_score=random.uniform(0.1, 0.9),
                vulnerability_score=random.uniform(0.1, 0.8),
                alternative_suppliers_count=random.randint(1, 10),
                substitution_difficulty=random.choice(["low", "medium", "high", "impossible"]),
                established_date=datetime.utcnow() - timedelta(days=random.randint(30, 3650)),
                last_transaction_date=datetime.utcnow() - timedelta(days=random.randint(1, 90))
            )
            relationships.append(relationship)
    
    # Create regional clusters
    for i in range(0, len(nodes), 20):
        cluster_nodes = nodes[i:i+20]
        if len(cluster_nodes) > 1:
            # Connect nodes within cluster
            for j in range(len(cluster_nodes)):
                for k in range(j+1, min(j+5, len(cluster_nodes))):  # Connect to up to 4 neighbors
                    if random.random() > 0.3:  # 70% chance of connection
                        relationship = SupplyChainRelationship(
                            upstream_node_id=cluster_nodes[j].id,
                            downstream_node_id=cluster_nodes[k].id,
                            relationship_type=random.choice(["supplier", "customer", "partner"]),
                            relationship_strength=random.uniform(0.2, 0.8),
                            trade_volume_usd=random.randint(1000, 5000000),
                            percentage_of_downstream_supply=random.uniform(0.01, 0.15),
                            criticality_score=random.uniform(0.1, 0.7),
                            vulnerability_score=random.uniform(0.1, 0.6),
                            alternative_suppliers_count=random.randint(2, 8),
                            substitution_difficulty=random.choice(["low", "medium", "high"]),
                            established_date=datetime.utcnow() - timedelta(days=random.randint(30, 2000)),
                            last_transaction_date=datetime.utcnow() - timedelta(days=random.randint(1, 60)),
                            created_at=datetime.utcnow()
                        )
                        relationships.append(relationship)
    
    session.add_all(relationships)
    session.commit()
    print(f"Generated {len(relationships)} supply chain relationships")
    return relationships

def main():
    """Generate all synthetic training data."""
    print("🔄 Starting synthetic training data generation...")
    
    # Create database tables
    from app.models import Base
    Base.metadata.create_all(bind=engine)
    
    # Create session
    session = SessionLocal()
    
    try:
        # Clear existing synthetic data
        print("\n0. Clearing existing synthetic data...")
        session.execute(text("DELETE FROM resilience_metrics"))
        session.execute(text("DELETE FROM sector_vulnerability_assessments"))
        session.execute(text("DELETE FROM acled_events"))
        session.execute(text("DELETE FROM cascade_events"))
        session.execute(text("DELETE FROM supply_chain_relationships"))
        session.execute(text("DELETE FROM supply_chain_nodes"))
        session.commit()
        print("Cleared existing data")
        
        # Generate data in dependency order
        print("\n1. Generating supply chain nodes...")
        nodes = generate_supply_chain_nodes(session, count=500)
        
        print("\n2. Generating supply chain relationships...")
        relationships = generate_supply_chain_relationships(session, nodes)
        
        print("\n3. Generating cascade events...")
        cascade_events = generate_cascade_events(session, nodes, count=1000)
        
        print("\n4. Generating ACLED events...")
        acled_events = generate_acled_events(session, count=2000)
        
        print("\n5. Generating vulnerability assessments...")
        assessments = generate_vulnerability_assessments(session, count=50)
        
        print("\n✅ Synthetic training data generation complete!")
        print(f"📊 Generated:")
        print(f"   - {len(nodes)} supply chain nodes")
        print(f"   - {len(relationships)} relationships")
        print(f"   - {len(cascade_events)} cascade events")
        print(f"   - {len(acled_events)} ACLED events")
        print(f"   - {len(assessments)} vulnerability assessments")
        
    except Exception as e:
        print(f"❌ Error generating synthetic data: {e}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":
    main()