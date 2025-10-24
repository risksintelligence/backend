"""
Network Analysis API endpoints for risk propagation and vulnerability assessment.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import logging
import json
import asyncio

from src.core.database import get_db
from src.core.dependencies import get_cache_manager
from src.cache.cache_manager import IntelligentCacheManager
from src.ml.models.network_analyzer import NetworkAnalyzer, PropagationResult
from src.data.models.network_models import NetworkNode, NetworkEdge, NetworkSnapshot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/network", tags=["network_analysis"])


async def get_network_analyzer() -> NetworkAnalyzer:
    """Get configured network analyzer instance."""
    return NetworkAnalyzer()


@router.get("/overview")
async def get_network_overview(
    db: AsyncSession = Depends(get_db),
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """
    Get network topology overview with connectivity metrics.
    
    Returns:
        - Network structure and connectivity metrics
        - Node count, edge count, density statistics  
        - Network visualization data
        - Critical component identification
    """
    try:
        cache_key = "network:overview"
        cached_data = await cache.get(cache_key, max_age_seconds=300)
        
        if cached_data:
            return {
                "status": "success",
                "data": cached_data,
                "source": "cache",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get network data from database
        nodes_result = await db.execute("SELECT * FROM network_nodes WHERE is_active = true")
        nodes_data = [dict(row) for row in nodes_result.fetchall()]
        
        edges_result = await db.execute("SELECT * FROM network_edges WHERE is_active = true")
        edges_data = [dict(row) for row in edges_result.fetchall()]
        
        if not nodes_data:
            # Require real network data - no sample data allowed
            raise HTTPException(
                status_code=503,
                detail="No real network data available - synthetic network data not allowed"
            )
        
        # Initialize network analyzer
        analyzer = await get_network_analyzer()
        analyzer.build_network_from_data(nodes_data, edges_data)
        
        # Calculate network metrics
        metrics = analyzer.calculate_network_metrics()
        
        # Get critical components
        critical_nodes = analyzer.identify_critical_nodes(threshold=0.7)
        vulnerabilities = analyzer.find_vulnerabilities()
        
        # Generate visualization data
        viz_data = analyzer.generate_network_visualization_data()
        
        overview_data = {
            "network_metrics": {
                "node_count": metrics.node_count,
                "edge_count": metrics.edge_count,
                "density": round(metrics.density, 4),
                "clustering_coefficient": round(metrics.clustering_coefficient, 4),
                "average_path_length": round(metrics.average_path_length, 2),
                "diameter": metrics.diameter,
                "connected_components": metrics.connected_components,
                "largest_component_size": metrics.largest_component_size
            },
            "critical_components": {
                "critical_nodes": critical_nodes[:10],  # Top 10
                "articulation_points": vulnerabilities["articulation_points"][:5],
                "bridge_edges": vulnerabilities["bridge_edges"][:5],
                "single_points_of_failure": vulnerabilities["single_points_of_failure"]
            },
            "visualization_data": viz_data,
            "health_indicators": {
                "resilience_score": calculate_resilience_score(metrics, vulnerabilities),
                "connectivity_health": "healthy" if metrics.density > 0.1 else "sparse",
                "vulnerability_level": "high" if len(vulnerabilities["single_points_of_failure"]) > 5 else "moderate"
            }
        }
        
        # Cache the results
        await cache.set(cache_key, overview_data, ttl_seconds=300)
        
        return {
            "status": "success",
            "data": overview_data,
            "source": "computed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in network overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes")
async def get_node_analysis(
    node_id: Optional[str] = Query(None, description="Specific node to analyze"),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """
    Get individual node analysis with centrality measures.
    
    Returns:
        - Node centrality measures (betweenness, closeness, eigenvector)
        - Node influence and importance scores
        - Local neighborhood analysis
        - Node risk profiles and vulnerabilities
    """
    try:
        cache_key = f"network:nodes:{node_id or 'all'}:{limit}"
        cached_data = await cache.get(cache_key, max_age_seconds=600)
        
        if cached_data:
            return {
                "status": "success",
                "data": cached_data,
                "source": "cache",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get network data
        nodes_result = await db.execute("SELECT * FROM network_nodes WHERE is_active = true LIMIT %s", (limit,))
        nodes_data = [dict(row) for row in nodes_result.fetchall()]
        
        edges_result = await db.execute("SELECT * FROM network_edges WHERE is_active = true")
        edges_data = [dict(row) for row in edges_result.fetchall()]
        
        if not nodes_data:
            # Require real network data - no sample data allowed
            raise HTTPException(
                status_code=503,
                detail="No real network data available - synthetic network data not allowed"
            )
        
        # Initialize analyzer
        analyzer = await get_network_analyzer()
        analyzer.build_network_from_data(nodes_data, edges_data)
        
        if node_id:
            # Analyze specific node
            node_analysis = analyzer.analyze_node(node_id)
            if not node_analysis:
                raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
            
            analysis_data = {
                "node_analysis": {
                    "node_id": node_analysis.node_id,
                    "centrality_measures": node_analysis.centrality_measures,
                    "vulnerability_score": round(node_analysis.vulnerability_score, 4),
                    "influence_score": round(node_analysis.influence_score, 4),
                    "local_clustering": round(node_analysis.local_clustering, 4),
                    "neighborhood_size": node_analysis.neighborhood_size,
                    "risk_level": node_analysis.risk_level
                }
            }
        else:
            # Analyze all nodes
            centrality_measures = analyzer.calculate_centrality_measures()
            
            nodes_analysis = []
            for node in list(centrality_measures.keys())[:limit]:
                node_analysis = analyzer.analyze_node(node)
                if node_analysis:
                    nodes_analysis.append({
                        "node_id": node_analysis.node_id,
                        "centrality_measures": node_analysis.centrality_measures,
                        "vulnerability_score": round(node_analysis.vulnerability_score, 4),
                        "influence_score": round(node_analysis.influence_score, 4),
                        "risk_level": node_analysis.risk_level
                    })
            
            analysis_data = {
                "nodes_analysis": nodes_analysis,
                "total_nodes": len(centrality_measures),
                "analyzed_nodes": len(nodes_analysis)
            }
        
        # Cache results
        await cache.set(cache_key, analysis_data, ttl_seconds=600)
        
        return {
            "status": "success",
            "data": analysis_data,
            "source": "computed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in node analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/centrality")
async def get_centrality_analysis(
    metric: str = Query("all", description="Centrality metric: betweenness, closeness, eigenvector, pagerank, or all"),
    top_n: int = Query(20, ge=5, le=100),
    cache: IntelligentCacheManager = Depends(get_cache_manager),
    db: AsyncSession = Depends(get_db)
):
    """
    Get network centrality metrics for identifying key nodes.
    
    Returns:
        - Betweenness centrality for bottleneck identification
        - Closeness centrality for accessibility analysis  
        - Eigenvector centrality for influence measurement
        - PageRank centrality for authority ranking
    """
    try:
        cache_key = f"network:centrality:{metric}:{top_n}"
        cached_data = await cache.get(cache_key, max_age_seconds=900)
        
        if cached_data:
            return {
                "status": "success",
                "data": cached_data,
                "source": "cache",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get network data
        nodes_result = await db.execute("SELECT * FROM network_nodes WHERE is_active = true")
        nodes_data = [dict(row) for row in nodes_result.fetchall()]
        
        edges_result = await db.execute("SELECT * FROM network_edges WHERE is_active = true")
        edges_data = [dict(row) for row in edges_result.fetchall()]
        
        if not nodes_data:
            # Require real network data - no sample data allowed
            raise HTTPException(
                status_code=503,
                detail="No real network data available - synthetic network data not allowed"
            )
        
        # Initialize analyzer
        analyzer = await get_network_analyzer()
        analyzer.build_network_from_data(nodes_data, edges_data)
        
        # Calculate centrality measures
        centrality_measures = analyzer.calculate_centrality_measures()
        
        if metric == "all":
            # Return all centrality measures for top nodes
            combined_scores = {}
            for node, measures in centrality_measures.items():
                combined_scores[node] = sum(measures.values()) / len(measures)
            
            top_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            centrality_data = {
                "all_centrality_measures": {
                    node: centrality_measures[node] for node, _ in top_nodes
                },
                "top_nodes_combined": [{"node_id": node, "combined_score": score} for node, score in top_nodes],
                "metric_summaries": {
                    "betweenness": get_metric_summary(centrality_measures, "betweenness"),
                    "closeness": get_metric_summary(centrality_measures, "closeness"),
                    "eigenvector": get_metric_summary(centrality_measures, "eigenvector"),
                    "pagerank": get_metric_summary(centrality_measures, "pagerank")
                }
            }
        else:
            # Return specific centrality metric
            if metric not in ["betweenness", "closeness", "eigenvector", "pagerank", "degree"]:
                raise HTTPException(status_code=400, detail=f"Invalid metric: {metric}")
            
            metric_scores = {node: measures.get(metric, 0) for node, measures in centrality_measures.items()}
            top_nodes = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            centrality_data = {
                f"{metric}_centrality": [{"node_id": node, "score": score} for node, score in top_nodes],
                "metric_summary": get_metric_summary(centrality_measures, metric),
                "interpretation": get_centrality_interpretation(metric)
            }
        
        # Cache results
        await cache.set(cache_key, centrality_data, ttl_seconds=900)
        
        return {
            "status": "success",
            "data": centrality_data,
            "source": "computed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in centrality analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vulnerabilities")
async def get_vulnerability_assessment(
    assessment_type: str = Query("comprehensive", description="Assessment type: comprehensive, spof, bridges"),
    cache: IntelligentCacheManager = Depends(get_cache_manager),
    db: AsyncSession = Depends(get_db)
):
    """
    Get network vulnerability assessment and weak points.
    
    Returns:
        - Single points of failure identification
        - Cascade failure risk analysis
        - Network resilience metrics
        - Critical edge and node vulnerability scores
    """
    try:
        cache_key = f"network:vulnerabilities:{assessment_type}"
        cached_data = await cache.get(cache_key, max_age_seconds=600)
        
        if cached_data:
            return {
                "status": "success",
                "data": cached_data,
                "source": "cache",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get network data
        nodes_result = await db.execute("SELECT * FROM network_nodes WHERE is_active = true")
        nodes_data = [dict(row) for row in nodes_result.fetchall()]
        
        edges_result = await db.execute("SELECT * FROM network_edges WHERE is_active = true")
        edges_data = [dict(row) for row in edges_result.fetchall()]
        
        if not nodes_data:
            # Require real network data - no sample data allowed
            raise HTTPException(
                status_code=503,
                detail="No real network data available - synthetic network data not allowed"
            )
        
        # Initialize analyzer
        analyzer = await get_network_analyzer()
        analyzer.build_network_from_data(nodes_data, edges_data)
        
        # Analyze vulnerabilities
        vulnerabilities = analyzer.find_vulnerabilities()
        
        # Calculate node vulnerability scores
        node_vulnerabilities = {}
        for node_data in nodes_data:
            node_id = node_data['node_id']
            node_analysis = analyzer.analyze_node(node_id)
            if node_analysis:
                node_vulnerabilities[node_id] = node_analysis.vulnerability_score
        
        if assessment_type == "comprehensive":
            vulnerability_data = {
                "overall_assessment": {
                    "vulnerability_level": calculate_overall_vulnerability_level(vulnerabilities, node_vulnerabilities),
                    "critical_vulnerabilities_count": len(vulnerabilities["single_points_of_failure"]),
                    "resilience_score": calculate_network_resilience(vulnerabilities, len(nodes_data))
                },
                "single_points_of_failure": {
                    "nodes": vulnerabilities["single_points_of_failure"],
                    "count": len(vulnerabilities["single_points_of_failure"]),
                    "impact_analysis": "High - These nodes are critical for network connectivity"
                },
                "structural_vulnerabilities": {
                    "articulation_points": vulnerabilities["articulation_points"],
                    "bridge_edges": vulnerabilities["bridge_edges"],
                    "vulnerable_components": vulnerabilities["vulnerable_components"]
                },
                "vulnerability_scores": {
                    "by_node": dict(sorted(node_vulnerabilities.items(), key=lambda x: x[1], reverse=True)[:20]),
                    "average_vulnerability": sum(node_vulnerabilities.values()) / len(node_vulnerabilities) if node_vulnerabilities else 0
                },
                "recommendations": generate_vulnerability_recommendations(vulnerabilities, node_vulnerabilities)
            }
        elif assessment_type == "spof":
            vulnerability_data = {
                "single_points_of_failure": vulnerabilities["single_points_of_failure"],
                "articulation_points": vulnerabilities["articulation_points"],
                "critical_analysis": analyze_spof_impact(vulnerabilities, analyzer),
                "mitigation_strategies": generate_spof_mitigation_strategies(vulnerabilities["single_points_of_failure"])
            }
        elif assessment_type == "bridges":
            vulnerability_data = {
                "bridge_edges": vulnerabilities["bridge_edges"],
                "bridge_analysis": analyze_bridge_criticality(vulnerabilities["bridge_edges"], analyzer),
                "redundancy_recommendations": generate_bridge_redundancy_recommendations(vulnerabilities["bridge_edges"])
            }
        else:
            raise HTTPException(status_code=400, detail=f"Invalid assessment type: {assessment_type}")
        
        # Cache results
        await cache.set(cache_key, vulnerability_data, ttl_seconds=600)
        
        return {
            "status": "success",
            "data": vulnerability_data,
            "source": "computed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in vulnerability assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/propagation")
async def get_risk_propagation_analysis(
    scenario: str = Query("default", description="Propagation scenario: default, high_impact, targeted"),
    cache: IntelligentCacheManager = Depends(get_cache_manager),
    db: AsyncSession = Depends(get_db)
):
    """
    Get risk propagation and contagion effect analysis.
    
    Returns:
        - Shock propagation simulation
        - Contagion effect modeling
        - Risk amplification pathways
        - Systemic risk assessment
    """
    try:
        cache_key = f"network:propagation:{scenario}"
        cached_data = await cache.get(cache_key, max_age_seconds=1800)
        
        if cached_data:
            return {
                "status": "success",
                "data": cached_data,
                "source": "cache",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get network data
        nodes_result = await db.execute("SELECT * FROM network_nodes WHERE is_active = true")
        nodes_data = [dict(row) for row in nodes_result.fetchall()]
        
        edges_result = await db.execute("SELECT * FROM network_edges WHERE is_active = true")
        edges_data = [dict(row) for row in edges_result.fetchall()]
        
        if not nodes_data:
            # Require real network data - no sample data allowed
            raise HTTPException(
                status_code=503,
                detail="No real network data available - synthetic network data not allowed"
            )
        
        # Initialize analyzer
        analyzer = await get_network_analyzer()
        analyzer.build_network_from_data(nodes_data, edges_data)
        
        # Define propagation scenarios
        if scenario == "default":
            initial_nodes = analyzer.identify_critical_nodes(threshold=0.8)[:2]
            shock_magnitude = 0.7
        elif scenario == "high_impact":
            initial_nodes = analyzer.identify_critical_nodes(threshold=0.9)[:1]
            shock_magnitude = 1.0
        elif scenario == "targeted":
            vulnerabilities = analyzer.find_vulnerabilities()
            initial_nodes = vulnerabilities["single_points_of_failure"][:3]
            shock_magnitude = 0.8
        else:
            raise HTTPException(status_code=400, detail=f"Invalid scenario: {scenario}")
        
        if not initial_nodes:
            initial_nodes = [nodes_data[0]['node_id']] if nodes_data else []
        
        # Run propagation simulation
        propagation_result = analyzer.simulate_risk_propagation(
            initial_nodes=initial_nodes,
            shock_magnitude=shock_magnitude,
            steps=30,
            containment_threshold=0.1
        )
        
        # Analyze propagation patterns
        propagation_analysis = analyze_propagation_patterns(propagation_result, analyzer)
        
        propagation_data = {
            "simulation_summary": {
                "simulation_id": propagation_result.simulation_id,
                "initial_nodes": propagation_result.initial_nodes,
                "affected_nodes_count": len(propagation_result.affected_nodes),
                "total_nodes": len(nodes_data),
                "propagation_percentage": round(len(propagation_result.affected_nodes) / len(nodes_data) * 100, 2),
                "recovery_estimate_hours": propagation_result.recovery_estimate
            },
            "propagation_dynamics": {
                "propagation_steps": propagation_result.propagation_steps[-10:],  # Last 10 steps
                "peak_infection_step": find_peak_infection_step(propagation_result.propagation_steps),
                "containment_achieved": len(propagation_result.propagation_steps) < 30
            },
            "impact_analysis": {
                "affected_nodes": propagation_result.affected_nodes,
                "final_impact": propagation_result.final_impact,
                "high_impact_nodes": [node for node, impact in propagation_result.final_impact.items() if impact > 0.5]
            },
            "systemic_risk_assessment": {
                "systemic_risk_level": calculate_systemic_risk_level(propagation_result, len(nodes_data)),
                "contagion_potential": calculate_contagion_potential(propagation_result),
                "network_stability": assess_network_stability(propagation_result, analyzer)
            },
            "containment_strategies": propagation_result.containment_strategies,
            "propagation_patterns": propagation_analysis
        }
        
        # Cache results
        await cache.set(cache_key, propagation_data, ttl_seconds=1800)
        
        return {
            "status": "success",
            "data": propagation_data,
            "source": "computed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in propagation analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/critical-paths")
async def get_critical_paths_analysis(
    source_node: str = Query(..., description="Source node for path analysis"),
    target_nodes: Optional[str] = Query(None, description="Comma-separated target nodes"),
    max_paths: int = Query(10, ge=1, le=50),
    cache: IntelligentCacheManager = Depends(get_cache_manager),
    db: AsyncSession = Depends(get_db)
):
    """
    Get critical path analysis for essential connections.
    
    Returns:
        - Shortest path analysis between key nodes
        - Alternative route identification
        - Path redundancy analysis
        - Critical infrastructure dependencies
    """
    try:
        target_list = target_nodes.split(',') if target_nodes else None
        cache_key = f"network:paths:{source_node}:{target_nodes or 'all'}:{max_paths}"
        cached_data = await cache.get(cache_key, max_age_seconds=1200)
        
        if cached_data:
            return {
                "status": "success",
                "data": cached_data,
                "source": "cache",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get network data
        nodes_result = await db.execute("SELECT * FROM network_nodes WHERE is_active = true")
        nodes_data = [dict(row) for row in nodes_result.fetchall()]
        
        edges_result = await db.execute("SELECT * FROM network_edges WHERE is_active = true")
        edges_data = [dict(row) for row in edges_result.fetchall()]
        
        if not nodes_data:
            # Require real network data - no sample data allowed
            raise HTTPException(
                status_code=503,
                detail="No real network data available - synthetic network data not allowed"
            )
        
        # Verify source node exists
        if not any(node['node_id'] == source_node for node in nodes_data):
            raise HTTPException(status_code=404, detail=f"Source node {source_node} not found")
        
        # Initialize analyzer
        analyzer = await get_network_analyzer()
        analyzer.build_network_from_data(nodes_data, edges_data)
        
        # Calculate shortest paths
        paths_analysis = analyzer.calculate_shortest_paths(source_node, target_list)
        
        if not paths_analysis:
            raise HTTPException(status_code=400, detail="No paths found from source node")
        
        # Analyze path criticality and redundancy
        critical_paths_data = analyze_path_criticality(paths_analysis, analyzer, max_paths)
        
        paths_data = {
            "source_node": source_node,
            "path_analysis": {
                "reachable_nodes": paths_analysis["reachable_nodes"],
                "unreachable_nodes": paths_analysis["unreachable_nodes"],
                "total_network_nodes": len(nodes_data)
            },
            "critical_paths": critical_paths_data["critical_paths"],
            "path_redundancy": critical_paths_data["redundancy_analysis"],
            "infrastructure_dependencies": critical_paths_data["infrastructure_deps"],
            "recommendations": generate_path_recommendations(critical_paths_data)
        }
        
        # Cache results
        await cache.set(cache_key, paths_data, ttl_seconds=1200)
        
        return {
            "status": "success",
            "data": paths_data,
            "source": "computed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in critical paths analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supply-chain")
async def get_supply_chain_network_analysis(
    db: AsyncSession = Depends(get_db),
    cache: IntelligentCacheManager = Depends(get_cache_manager)
):
    """
    Get supply chain network analysis and risk assessment.
    
    Returns:
        - Supply chain network topology
        - Supplier dependency analysis
        - Risk concentration metrics
        - Critical supplier identification
        - Supply chain resilience assessment
    """
    try:
        cache_key = "network:supply_chain"
        cached_data = await cache.get(cache_key, max_age_seconds=600)
        
        if cached_data:
            return {
                "status": "success",
                "data": cached_data,
                "source": "cache",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get supply chain network data
        nodes_result = await db.execute(
            "SELECT * FROM network_nodes WHERE node_type = 'supplier' AND is_active = true"
        )
        supplier_nodes = [dict(row) for row in nodes_result.fetchall()]
        
        edges_result = await db.execute(
            "SELECT * FROM network_edges WHERE edge_type = 'supply_chain' AND is_active = true"
        )
        supply_edges = [dict(row) for row in edges_result.fetchall()]
        
        if not supplier_nodes:
            # Use real economic data to assess supply chain risks when no direct data available
            from src.data.sources import fred
            
            # Get economic indicators that affect supply chains
            results = await asyncio.gather(
                fred.get_economic_stability_indicators(),
                fred.get_market_volatility_indicators(),
                fred.get_treasury_yields(),
                return_exceptions=True
            )
            
            supply_chain_data = {
                "supply_chain_status": "assessment_based_on_economic_indicators",
                "economic_risk_factors": {},
                "overall_risk_score": 50,  # Neutral baseline
                "risk_assessment": {
                    "economic_stability": "monitoring",
                    "market_volatility": "normal",
                    "interest_rate_environment": "stable"
                },
                "recommendations": [
                    "Implement supply chain monitoring system",
                    "Establish supplier diversity metrics",
                    "Monitor economic indicators for supply chain impacts"
                ],
                "data_availability": "limited - economic indicators used for assessment"
            }
            
            # Process economic indicators
            source_names = ["stability", "volatility", "yields"]
            for i, result in enumerate(results):
                if isinstance(result, dict) and result:
                    supply_chain_data["economic_risk_factors"][source_names[i]] = result
            
            # Calculate risk score based on economic indicators
            if supply_chain_data["economic_risk_factors"]:
                risk_factors = len(supply_chain_data["economic_risk_factors"])
                supply_chain_data["overall_risk_score"] = min(100, 40 + (risk_factors * 10))
            
            await cache.set(cache_key, supply_chain_data, ttl_seconds=600)
            
            return {
                "status": "success",
                "data": supply_chain_data,
                "source": "computed_from_economic_indicators",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Initialize network analyzer with supply chain data
        analyzer = await get_network_analyzer()
        analyzer.build_network_from_data(supplier_nodes, supply_edges)
        
        # Calculate supply chain metrics
        metrics = analyzer.calculate_network_metrics()
        vulnerabilities = analyzer.find_vulnerabilities()
        critical_suppliers = analyzer.identify_critical_nodes(threshold=0.6)
        
        # Supply chain specific analysis
        supply_chain_data = {
            "network_structure": {
                "total_suppliers": metrics.node_count,
                "supply_relationships": metrics.edge_count,
                "network_density": round(metrics.density, 4),
                "supply_chain_complexity": round(metrics.average_path_length, 2)
            },
            "critical_suppliers": {
                "tier_1_critical": critical_suppliers[:5],
                "single_source_dependencies": vulnerabilities["single_points_of_failure"],
                "supplier_concentration_risk": calculate_supplier_concentration(supplier_nodes)
            },
            "risk_assessment": {
                "overall_risk_score": calculate_supply_chain_risk_score(metrics, vulnerabilities),
                "dependency_risk": "high" if len(vulnerabilities["single_points_of_failure"]) > 3 else "medium",
                "diversification_level": calculate_diversification_level(supplier_nodes),
                "resilience_score": calculate_resilience_score(metrics, vulnerabilities)
            },
            "vulnerability_analysis": {
                "geographic_concentration": analyze_geographic_concentration(supplier_nodes),
                "industry_concentration": analyze_industry_concentration(supplier_nodes),
                "tier_dependency": analyze_tier_dependencies(supplier_nodes, supply_edges)
            },
            "recommendations": generate_supply_chain_recommendations(vulnerabilities, supplier_nodes)
        }
        
        # Cache the results
        await cache.set(cache_key, supply_chain_data, ttl_seconds=600)
        
        return {
            "status": "success",
            "data": supply_chain_data,
            "source": "computed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in supply chain analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulation")
async def run_shock_simulation(
    simulation_request: Dict[str, Any],
    cache: IntelligentCacheManager = Depends(get_cache_manager),
    db: AsyncSession = Depends(get_db)
):
    """
    Run custom shock simulation for impact modeling.
    
    Body:
        - initial_nodes: List of nodes to apply shock
        - shock_magnitude: Magnitude of initial shock (0-1)
        - simulation_steps: Number of simulation steps
        - containment_threshold: Threshold for containment
    
    Returns:
        - Node failure simulation results
        - Edge disruption modeling
        - Cascading failure analysis
        - Recovery time estimation
    """
    try:
        # Validate request
        initial_nodes = simulation_request.get('initial_nodes', [])
        shock_magnitude = simulation_request.get('shock_magnitude', 0.5)
        simulation_steps = simulation_request.get('simulation_steps', 50)
        containment_threshold = simulation_request.get('containment_threshold', 0.1)
        
        if not initial_nodes:
            raise HTTPException(status_code=400, detail="initial_nodes is required")
        
        if not 0 <= shock_magnitude <= 1:
            raise HTTPException(status_code=400, detail="shock_magnitude must be between 0 and 1")
        
        # Get network data
        nodes_result = await db.execute("SELECT * FROM network_nodes WHERE is_active = true")
        nodes_data = [dict(row) for row in nodes_result.fetchall()]
        
        edges_result = await db.execute("SELECT * FROM network_edges WHERE is_active = true")
        edges_data = [dict(row) for row in edges_result.fetchall()]
        
        if not nodes_data:
            # Require real network data - no sample data allowed
            raise HTTPException(
                status_code=503,
                detail="No real network data available - synthetic network data not allowed"
            )
        
        # Verify all initial nodes exist
        available_nodes = {node['node_id'] for node in nodes_data}
        invalid_nodes = set(initial_nodes) - available_nodes
        if invalid_nodes:
            raise HTTPException(status_code=400, detail=f"Invalid nodes: {invalid_nodes}")
        
        # Initialize analyzer
        analyzer = await get_network_analyzer()
        analyzer.build_network_from_data(nodes_data, edges_data)
        
        # Run simulation
        simulation_result = analyzer.simulate_risk_propagation(
            initial_nodes=initial_nodes,
            shock_magnitude=shock_magnitude,
            steps=simulation_steps,
            containment_threshold=containment_threshold
        )
        
        # Analyze cascade effects
        cascade_analysis = analyze_cascade_effects(simulation_result, analyzer)
        
        # Calculate recovery metrics
        recovery_analysis = calculate_recovery_metrics(simulation_result, nodes_data)
        
        simulation_data = {
            "simulation_metadata": {
                "simulation_id": simulation_result.simulation_id,
                "parameters": {
                    "initial_nodes": initial_nodes,
                    "shock_magnitude": shock_magnitude,
                    "simulation_steps": simulation_steps,
                    "containment_threshold": containment_threshold
                },
                "execution_timestamp": datetime.utcnow().isoformat()
            },
            "simulation_results": {
                "total_affected_nodes": len(simulation_result.affected_nodes),
                "propagation_completed_steps": len(simulation_result.propagation_steps),
                "final_impact_summary": {
                    "high_impact": len([n for n, i in simulation_result.final_impact.items() if i > 0.7]),
                    "medium_impact": len([n for n, i in simulation_result.final_impact.items() if 0.3 <= i <= 0.7]),
                    "low_impact": len([n for n, i in simulation_result.final_impact.items() if 0 < i < 0.3])
                }
            },
            "cascade_analysis": cascade_analysis,
            "recovery_analysis": recovery_analysis,
            "containment_strategies": simulation_result.containment_strategies,
            "recommendations": generate_simulation_recommendations(simulation_result, cascade_analysis)
        }
        
        return {
            "status": "success",
            "data": simulation_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in shock simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions



def calculate_resilience_score(metrics: Any, vulnerabilities: Dict) -> float:
    """Calculate network resilience score."""
    base_score = metrics.density * 50 + metrics.clustering_coefficient * 30
    vulnerability_penalty = len(vulnerabilities["single_points_of_failure"]) * 10
    return max(0, min(100, base_score - vulnerability_penalty))


def get_metric_summary(centrality_measures: Dict, metric: str) -> Dict:
    """Get statistical summary for a centrality metric."""
    values = [measures.get(metric, 0) for measures in centrality_measures.values()]
    return {
        "max": max(values) if values else 0,
        "min": min(values) if values else 0,
        "mean": sum(values) / len(values) if values else 0,
        "std": np.std(values) if values else 0
    }


def get_centrality_interpretation(metric: str) -> str:
    """Get interpretation for centrality metric."""
    interpretations = {
        "betweenness": "Identifies bottleneck nodes that control information flow",
        "closeness": "Measures how quickly a node can reach all other nodes",
        "eigenvector": "Identifies nodes connected to other important nodes",
        "pagerank": "Ranks nodes by their authority and influence",
        "degree": "Shows nodes with the most direct connections"
    }
    return interpretations.get(metric, "Centrality measure for network analysis")


def calculate_overall_vulnerability_level(vulnerabilities: Dict, node_vulnerabilities: Dict) -> str:
    """Calculate overall vulnerability level."""
    spof_count = len(vulnerabilities["single_points_of_failure"])
    avg_vulnerability = sum(node_vulnerabilities.values()) / len(node_vulnerabilities) if node_vulnerabilities else 0
    
    if spof_count > 5 or avg_vulnerability > 0.7:
        return "critical"
    elif spof_count > 2 or avg_vulnerability > 0.5:
        return "high"
    elif spof_count > 0 or avg_vulnerability > 0.3:
        return "medium"
    else:
        return "low"


def calculate_network_resilience(vulnerabilities: Dict, node_count: int) -> float:
    """Calculate network resilience score."""
    spof_ratio = len(vulnerabilities["single_points_of_failure"]) / node_count if node_count > 0 else 1
    bridge_ratio = len(vulnerabilities["bridge_edges"]) / node_count if node_count > 0 else 1
    
    resilience = 1.0 - (spof_ratio * 0.6 + bridge_ratio * 0.4)
    return max(0, min(1, resilience))


def generate_vulnerability_recommendations(vulnerabilities: Dict, node_vulnerabilities: Dict) -> List[str]:
    """Generate vulnerability mitigation recommendations."""
    recommendations = []
    
    if vulnerabilities["single_points_of_failure"]:
        recommendations.append(f"Implement redundancy for {len(vulnerabilities['single_points_of_failure'])} critical nodes")
    
    if vulnerabilities["bridge_edges"]:
        recommendations.append("Add alternative connections to eliminate bridge dependencies")
    
    high_vuln_nodes = [node for node, score in node_vulnerabilities.items() if score > 0.8]
    if high_vuln_nodes:
        recommendations.append(f"Strengthen {len(high_vuln_nodes)} highly vulnerable nodes")
    
    recommendations.append("Implement network monitoring and early warning systems")
    recommendations.append("Develop incident response procedures for network disruptions")
    
    return recommendations


def analyze_spof_impact(vulnerabilities: Dict, analyzer: NetworkAnalyzer) -> Dict:
    """Analyze impact of single points of failure."""
    spof_nodes = vulnerabilities["single_points_of_failure"]
    
    impact_analysis = {
        "total_spof": len(spof_nodes),
        "high_impact_spof": [],
        "network_fragmentation_risk": "high" if len(spof_nodes) > 3 else "medium"
    }
    
    # Analyze each SPOF
    for node in spof_nodes[:5]:  # Analyze top 5
        node_analysis = analyzer.analyze_node(node)
        if node_analysis and node_analysis.influence_score > 0.7:
            impact_analysis["high_impact_spof"].append({
                "node_id": node,
                "influence_score": node_analysis.influence_score,
                "vulnerability_score": node_analysis.vulnerability_score
            })
    
    return impact_analysis


def generate_spof_mitigation_strategies(spof_nodes: List[str]) -> List[str]:
    """Generate SPOF mitigation strategies."""
    strategies = [
        "Implement automated failover systems for critical nodes",
        "Create backup communication channels",
        "Establish redundant processing capabilities",
        "Deploy distributed architecture to reduce dependencies"
    ]
    
    if len(spof_nodes) > 5:
        strategies.append("Priority: Reduce network centralization")
    
    return strategies


def analyze_bridge_criticality(bridge_edges: List[str], analyzer: NetworkAnalyzer) -> Dict:
    """Analyze criticality of bridge edges."""
    return {
        "total_bridges": len(bridge_edges),
        "criticality_level": "high" if len(bridge_edges) > 3 else "medium",
        "redundancy_gaps": len(bridge_edges),
        "connectivity_risk": "Network connectivity depends on critical bridges"
    }


def generate_bridge_redundancy_recommendations(bridge_edges: List[str]) -> List[str]:
    """Generate bridge redundancy recommendations."""
    return [
        "Add alternative pathways to bypass critical bridges",
        "Implement mesh connectivity where possible",
        "Monitor bridge edge health continuously",
        "Establish emergency rerouting protocols"
    ]


def analyze_propagation_patterns(propagation_result: PropagationResult, analyzer: NetworkAnalyzer) -> Dict:
    """Analyze propagation patterns from simulation."""
    return {
        "propagation_speed": len(propagation_result.propagation_steps),
        "infection_pattern": "cascading" if len(propagation_result.affected_nodes) > len(propagation_result.initial_nodes) * 3 else "contained",
        "amplification_detected": any(step["total_risk"] > len(propagation_result.initial_nodes) for step in propagation_result.propagation_steps),
        "containment_effectiveness": "effective" if len(propagation_result.propagation_steps) < 20 else "limited"
    }


def find_peak_infection_step(propagation_steps: List[Dict]) -> int:
    """Find the step with peak infection."""
    if not propagation_steps:
        return 0
    
    max_risk_step = max(propagation_steps, key=lambda x: x.get("total_risk", 0))
    return max_risk_step.get("step", 0)


def calculate_systemic_risk_level(propagation_result: PropagationResult, total_nodes: int) -> str:
    """Calculate systemic risk level."""
    affected_ratio = len(propagation_result.affected_nodes) / total_nodes if total_nodes > 0 else 0
    
    if affected_ratio > 0.7:
        return "critical"
    elif affected_ratio > 0.4:
        return "high"
    elif affected_ratio > 0.2:
        return "medium"
    else:
        return "low"


def calculate_contagion_potential(propagation_result: PropagationResult) -> float:
    """Calculate contagion potential score."""
    initial_count = len(propagation_result.initial_nodes)
    affected_count = len(propagation_result.affected_nodes)
    
    if initial_count == 0:
        return 0.0
    
    contagion_ratio = affected_count / initial_count
    return min(1.0, contagion_ratio / 10)  # Normalize to 0-1


def assess_network_stability(propagation_result: PropagationResult, analyzer: NetworkAnalyzer) -> str:
    """Assess overall network stability."""
    if len(propagation_result.affected_nodes) > 10:
        return "unstable"
    elif len(propagation_result.propagation_steps) > 30:
        return "fragile"
    else:
        return "stable"


def analyze_path_criticality(paths_analysis: Dict, analyzer: NetworkAnalyzer, max_paths: int) -> Dict:
    """Analyze path criticality and redundancy."""
    critical_paths = []
    
    for target, path_info in list(paths_analysis["paths"].items())[:max_paths]:
        critical_paths.append({
            "target": target,
            "path": path_info["path"],
            "length": path_info["length"],
            "criticality": path_info["criticality"],
            "hop_count": len(path_info["path"]) - 1
        })
    
    return {
        "critical_paths": critical_paths,
        "redundancy_analysis": {
            "single_path_dependencies": len([p for p in critical_paths if p["hop_count"] == 1]),
            "multi_hop_paths": len([p for p in critical_paths if p["hop_count"] > 1]),
            "high_criticality_paths": len([p for p in critical_paths if p["criticality"] > 0.7])
        },
        "infrastructure_deps": {
            "critical_intermediate_nodes": analyze_intermediate_nodes(critical_paths),
            "path_diversity_score": calculate_path_diversity(critical_paths)
        }
    }


def analyze_intermediate_nodes(critical_paths: List[Dict]) -> List[str]:
    """Analyze intermediate nodes in critical paths."""
    intermediate_counts = {}
    
    for path_info in critical_paths:
        path = path_info["path"]
        # Count intermediate nodes (exclude source and target)
        for node in path[1:-1]:
            intermediate_counts[node] = intermediate_counts.get(node, 0) + 1
    
    # Return nodes that appear in multiple paths
    return [node for node, count in intermediate_counts.items() if count > 1]


def calculate_path_diversity(critical_paths: List[Dict]) -> float:
    """Calculate path diversity score."""
    if not critical_paths:
        return 0.0
    
    all_nodes = set()
    for path_info in critical_paths:
        all_nodes.update(path_info["path"])
    
    total_path_length = sum(len(path_info["path"]) for path_info in critical_paths)
    unique_nodes = len(all_nodes)
    
    # Diversity score: higher when paths use more unique nodes
    return unique_nodes / total_path_length if total_path_length > 0 else 0.0


def generate_path_recommendations(critical_paths_data: Dict) -> List[str]:
    """Generate path optimization recommendations."""
    recommendations = []
    
    if critical_paths_data["redundancy_analysis"]["single_path_dependencies"] > 0:
        recommendations.append("Add redundant pathways for single-hop dependencies")
    
    if critical_paths_data["infrastructure_deps"]["path_diversity_score"] < 0.5:
        recommendations.append("Increase path diversity to reduce bottleneck risks")
    
    recommendations.extend([
        "Monitor critical intermediate nodes for performance",
        "Implement load balancing across multiple paths",
        "Establish backup routing protocols"
    ])
    
    return recommendations


def analyze_cascade_effects(simulation_result: PropagationResult, analyzer: NetworkAnalyzer) -> Dict:
    """Analyze cascade effects from simulation."""
    return {
        "cascade_magnitude": len(simulation_result.affected_nodes) / len(simulation_result.initial_nodes) if simulation_result.initial_nodes else 0,
        "cascade_depth": len(simulation_result.propagation_steps),
        "amplification_factor": calculate_amplification_factor(simulation_result),
        "cascade_pattern": determine_cascade_pattern(simulation_result)
    }


def calculate_amplification_factor(simulation_result: PropagationResult) -> float:
    """Calculate risk amplification factor."""
    if not simulation_result.propagation_steps:
        return 1.0
    
    initial_risk = simulation_result.propagation_steps[0].get("total_risk", 0)
    peak_risk = max(step.get("total_risk", 0) for step in simulation_result.propagation_steps)
    
    return peak_risk / initial_risk if initial_risk > 0 else 1.0


def determine_cascade_pattern(simulation_result: PropagationResult) -> str:
    """Determine the pattern of cascade propagation."""
    steps = simulation_result.propagation_steps
    if not steps:
        return "no_cascade"
    
    risk_progression = [step.get("total_risk", 0) for step in steps]
    
    if len(risk_progression) < 3:
        return "limited"
    
    # Check if risk keeps growing
    if risk_progression[-1] > risk_progression[0] * 1.5:
        return "exponential"
    elif risk_progression[-1] > risk_progression[0]:
        return "linear"
    else:
        return "contained"


def calculate_recovery_metrics(simulation_result: PropagationResult, nodes_data: List[Dict]) -> Dict:
    """Calculate recovery time and effort metrics."""
    affected_count = len(simulation_result.affected_nodes)
    total_count = len(nodes_data)
    
    return {
        "recovery_time_estimate": simulation_result.recovery_estimate,
        "affected_percentage": round(affected_count / total_count * 100, 2),
        "recovery_complexity": "high" if affected_count > total_count * 0.5 else "medium",
        "priority_recovery_nodes": simulation_result.affected_nodes[:5],  # Top 5 priority
        "resource_requirements": estimate_recovery_resources(affected_count)
    }


def estimate_recovery_resources(affected_count: int) -> Dict:
    """Estimate recovery resources needed."""
    return {
        "personnel_hours": affected_count * 8,
        "estimated_cost": affected_count * 10000,
        "priority_level": "critical" if affected_count > 10 else "high"
    }


def generate_simulation_recommendations(simulation_result: PropagationResult, cascade_analysis: Dict) -> List[str]:
    """Generate recommendations based on simulation results."""
    recommendations = []
    
    if cascade_analysis["cascade_pattern"] == "exponential":
        recommendations.append("Implement immediate circuit breakers to prevent exponential spread")
    
    if cascade_analysis["amplification_factor"] > 2.0:
        recommendations.append("Review and adjust amplification factors in network connections")
    
    if len(simulation_result.affected_nodes) > 15:
        recommendations.append("Consider network segmentation to limit impact scope")
    
    recommendations.extend([
        "Establish real-time monitoring for early cascade detection",
        "Develop automated containment protocols",
        "Create emergency response procedures for high-impact scenarios"
    ])
    
    return recommendations


# Supply chain helper functions

def calculate_supplier_concentration(supplier_nodes: List[Dict]) -> str:
    """Calculate supplier concentration risk level."""
    if len(supplier_nodes) < 5:
        return "high"
    elif len(supplier_nodes) < 15:
        return "medium"
    else:
        return "low"


def calculate_supply_chain_risk_score(metrics: Any, vulnerabilities: Dict) -> float:
    """Calculate overall supply chain risk score."""
    base_score = 50  # Neutral baseline
    
    # Adjust for network density (higher density = lower risk)
    density_adjustment = (metrics.density - 0.1) * 50
    
    # Adjust for vulnerabilities (more SPOFs = higher risk)
    spof_penalty = len(vulnerabilities["single_points_of_failure"]) * 10
    
    risk_score = base_score - density_adjustment + spof_penalty
    return max(0, min(100, risk_score))


def calculate_diversification_level(supplier_nodes: List[Dict]) -> str:
    """Calculate supplier diversification level."""
    unique_types = len(set(node.get('node_subtype', 'unknown') for node in supplier_nodes))
    
    if unique_types < 3:
        return "low"
    elif unique_types < 7:
        return "medium"
    else:
        return "high"


def analyze_geographic_concentration(supplier_nodes: List[Dict]) -> Dict:
    """Analyze geographic concentration of suppliers."""
    regions = {}
    for node in supplier_nodes:
        region = node.get('region', 'unknown')
        regions[region] = regions.get(region, 0) + 1
    
    total_suppliers = len(supplier_nodes)
    max_concentration = max(regions.values()) if regions else 0
    concentration_ratio = max_concentration / total_suppliers if total_suppliers > 0 else 0
    
    return {
        "regional_distribution": regions,
        "max_concentration_ratio": round(concentration_ratio, 3),
        "risk_level": "high" if concentration_ratio > 0.5 else "medium" if concentration_ratio > 0.3 else "low"
    }


def analyze_industry_concentration(supplier_nodes: List[Dict]) -> Dict:
    """Analyze industry concentration of suppliers."""
    industries = {}
    for node in supplier_nodes:
        industry = node.get('industry', 'unknown')
        industries[industry] = industries.get(industry, 0) + 1
    
    total_suppliers = len(supplier_nodes)
    max_concentration = max(industries.values()) if industries else 0
    concentration_ratio = max_concentration / total_suppliers if total_suppliers > 0 else 0
    
    return {
        "industry_distribution": industries,
        "max_concentration_ratio": round(concentration_ratio, 3),
        "risk_level": "high" if concentration_ratio > 0.4 else "medium" if concentration_ratio > 0.25 else "low"
    }


def analyze_tier_dependencies(supplier_nodes: List[Dict], supply_edges: List[Dict]) -> Dict:
    """Analyze supplier tier dependencies."""
    tier_counts = {}
    for node in supplier_nodes:
        tier = node.get('supplier_tier', 'unknown')
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    return {
        "tier_distribution": tier_counts,
        "tier_1_percentage": round(tier_counts.get('tier_1', 0) / len(supplier_nodes) * 100, 1) if supplier_nodes else 0,
        "dependency_depth": len(tier_counts),
        "risk_assessment": "balanced" if len(tier_counts) > 2 else "concentrated"
    }


def generate_supply_chain_recommendations(vulnerabilities: Dict, supplier_nodes: List[Dict]) -> List[str]:
    """Generate supply chain specific recommendations."""
    recommendations = []
    
    if len(vulnerabilities["single_points_of_failure"]) > 0:
        recommendations.append("Implement dual sourcing for critical single-source suppliers")
    
    if len(supplier_nodes) < 10:
        recommendations.append("Expand supplier base to reduce concentration risk")
    
    recommendations.extend([
        "Establish supplier performance monitoring system",
        "Develop contingency plans for critical supplier disruptions",
        "Implement regular supplier risk assessments",
        "Create supplier diversity and inclusion programs",
        "Establish strategic supplier partnerships for resilience"
    ])
    
    return recommendations