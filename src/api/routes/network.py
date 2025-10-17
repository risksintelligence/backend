"""
Network analysis API endpoints for systemic risk propagation.
"""
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from src.ml.models.network_analyzer import RiskNetworkAnalyzer, NetworkNode, NetworkEdge

router = APIRouter()


class NetworkNodeResponse(BaseModel):
    """Network node response model."""
    id: str
    name: str
    category: str
    riskLevel: float = Field(..., alias="risk_level")
    systemicImportance: float = Field(..., alias="systemic_importance")
    description: str

    class Config:
        allow_population_by_field_name = True


class NetworkLinkResponse(BaseModel):
    """Network link response model."""
    source: str
    target: str
    strength: float
    riskPropagation: float = Field(..., alias="risk_propagation")
    type: str = Field(..., alias="connection_type")
    description: str

    class Config:
        allow_population_by_field_name = True


class NetworkAnalysisResponse(BaseModel):
    """Network analysis response model."""
    nodes: List[NetworkNodeResponse]
    links: List[NetworkLinkResponse]
    centralityMetrics: Dict[str, float] = Field(..., alias="centrality_metrics")
    vulnerabilityScore: float = Field(..., alias="vulnerability_score")
    systemicRiskScore: float = Field(..., alias="systemic_risk_score")
    criticalPaths: List[List[str]] = Field(..., alias="critical_paths")
    timestamp: datetime
    metadata: Dict[str, Any]

    class Config:
        allow_population_by_field_name = True
        json_schema_extra = {
            "example": {
                "nodes": [
                    {
                        "id": "fed",
                        "name": "Federal Reserve",
                        "category": "government",
                        "riskLevel": 25.0,
                        "systemicImportance": 0.95,
                        "description": "Central banking system"
                    }
                ],
                "links": [
                    {
                        "source": "fed",
                        "target": "jpmorgan",
                        "strength": 0.85,
                        "riskPropagation": 0.75,
                        "type": "regulatory",
                        "description": "Monetary policy transmission"
                    }
                ],
                "centralityMetrics": {"fed": 0.85, "jpmorgan": 0.72},
                "vulnerabilityScore": 42.5,
                "systemicRiskScore": 38.7,
                "criticalPaths": [["china_trade", "apple", "jpmorgan"]],
                "timestamp": "2024-01-15T10:30:00",
                "metadata": {
                    "totalNodes": 12,
                    "totalLinks": 23,
                    "networkDensity": 0.35
                }
            }
        }


class ShockSimulationRequest(BaseModel):
    """Request model for shock simulation."""
    shockedNode: str = Field(..., alias="shocked_node")
    shockMagnitude: float = Field(..., ge=0, le=100, alias="shock_magnitude")

    class Config:
        allow_population_by_field_name = True


class ShockSimulationResponse(BaseModel):
    """Response model for shock simulation."""
    originalRisks: Dict[str, float] = Field(..., alias="original_risks")
    simulatedRisks: Dict[str, float] = Field(..., alias="simulated_risks")
    riskChanges: Dict[str, float] = Field(..., alias="risk_changes")
    shockedNode: str = Field(..., alias="shocked_node")
    shockMagnitude: float = Field(..., alias="shock_magnitude")
    timestamp: datetime

    class Config:
        allow_population_by_field_name = True


@router.get("/analysis", response_model=NetworkAnalysisResponse)
async def get_network_analysis():
    """
    Get comprehensive network analysis of systemic risk.
    
    Returns network structure, centrality metrics, and risk propagation analysis.
    """
    try:
        analyzer = RiskNetworkAnalyzer()
        analysis = analyzer.analyze_network()
        
        # Convert nodes to response format
        nodes = [
            NetworkNodeResponse(
                id=node.id,
                name=node.name,
                category=node.category,
                risk_level=node.risk_level,
                systemic_importance=node.systemic_importance,
                description=node.description
            )
            for node in analysis.nodes
        ]
        
        # Convert edges to response format
        links = [
            NetworkLinkResponse(
                source=edge.source,
                target=edge.target,
                strength=edge.strength,
                risk_propagation=edge.risk_propagation,
                connection_type=edge.connection_type,
                description=edge.description
            )
            for edge in analysis.edges
        ]
        
        # Calculate network density
        total_possible_links = len(analysis.nodes) * (len(analysis.nodes) - 1)
        network_density = len(analysis.edges) / total_possible_links if total_possible_links > 0 else 0
        
        return NetworkAnalysisResponse(
            nodes=nodes,
            links=links,
            centrality_metrics=analysis.centrality_metrics,
            vulnerability_score=analysis.vulnerability_score,
            systemic_risk_score=analysis.systemic_risk_score,
            critical_paths=analysis.critical_paths,
            timestamp=analysis.timestamp,
            metadata={
                "totalNodes": len(analysis.nodes),
                "totalLinks": len(analysis.edges),
                "networkDensity": round(network_density, 3),
                "analysisVersion": "1.0"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing network analysis: {str(e)}"
        )


@router.post("/simulate-shock", response_model=ShockSimulationResponse)
async def simulate_shock(request: ShockSimulationRequest):
    """
    Simulate a risk shock propagating through the network.
    
    Args:
        request: Shock simulation parameters including node and magnitude
        
    Returns:
        Simulation results showing risk propagation effects
    """
    try:
        analyzer = RiskNetworkAnalyzer()
        analysis = analyzer.analyze_network()
        
        # Get original risk levels
        original_risks = {node.id: node.risk_level for node in analysis.nodes}
        
        # Validate shocked node exists
        if request.shockedNode not in original_risks:
            raise HTTPException(
                status_code=400,
                detail=f"Node '{request.shockedNode}' not found in network"
            )
        
        # Run shock simulation
        simulated_risks = analyzer.simulate_shock(
            request.shockedNode, 
            request.shockMagnitude
        )
        
        # Calculate risk changes
        risk_changes = {
            node_id: simulated_risks[node_id] - original_risks[node_id]
            for node_id in original_risks.keys()
        }
        
        return ShockSimulationResponse(
            original_risks=original_risks,
            simulated_risks=simulated_risks,
            risk_changes=risk_changes,
            shocked_node=request.shockedNode,
            shock_magnitude=request.shockMagnitude,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running shock simulation: {str(e)}"
        )


@router.get("/centrality")
async def get_centrality_metrics():
    """
    Get node centrality metrics for network importance analysis.
    
    Returns centrality scores for each node in the network.
    """
    try:
        analyzer = RiskNetworkAnalyzer()
        analysis = analyzer.analyze_network()
        
        # Enhanced centrality analysis
        centrality_data = []
        for node in analysis.nodes:
            centrality_score = analysis.centrality_metrics.get(node.id, 0)
            centrality_data.append({
                "nodeId": node.id,
                "nodeName": node.name,
                "category": node.category,
                "centralityScore": round(centrality_score, 4),
                "systemicImportance": node.systemic_importance,
                "riskLevel": node.risk_level,
                "combinedScore": round((centrality_score + node.systemic_importance) / 2, 4)
            })
        
        # Sort by combined score
        centrality_data.sort(key=lambda x: x["combinedScore"], reverse=True)
        
        return {
            "centrality": centrality_data,
            "topNodes": centrality_data[:5],
            "analysisTimestamp": datetime.utcnow(),
            "methodology": {
                "description": "Combined centrality and systemic importance scoring",
                "factors": [
                    "Degree centrality (30%)",
                    "Betweenness centrality (30%)",
                    "Eigenvector centrality (20%)",
                    "PageRank score (20%)"
                ]
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating centrality metrics: {str(e)}"
        )


@router.get("/critical-paths")
async def get_critical_paths():
    """
    Get critical risk propagation paths in the network.
    
    Returns paths that represent the highest risk propagation routes.
    """
    try:
        analyzer = RiskNetworkAnalyzer()
        analysis = analyzer.analyze_network()
        
        # Enhanced critical path analysis
        paths_data = []
        for i, path in enumerate(analysis.critical_paths):
            # Calculate path risk score
            path_risk = 0
            path_details = []
            
            for j in range(len(path)):
                node = next((n for n in analysis.nodes if n.id == path[j]), None)
                if node:
                    path_details.append({
                        "nodeId": node.id,
                        "nodeName": node.name,
                        "category": node.category,
                        "riskLevel": node.risk_level
                    })
                    path_risk += node.risk_level
            
            avg_path_risk = path_risk / len(path) if path else 0
            
            paths_data.append({
                "pathId": i + 1,
                "path": path,
                "pathDetails": path_details,
                "avgRiskLevel": round(avg_path_risk, 2),
                "pathLength": len(path),
                "riskCategory": (
                    "Critical" if avg_path_risk > 60 else
                    "High" if avg_path_risk > 40 else
                    "Moderate"
                )
            })
        
        return {
            "criticalPaths": paths_data,
            "totalPaths": len(paths_data),
            "analysisTimestamp": datetime.utcnow(),
            "methodology": {
                "description": "Critical paths from high-risk to systemically important nodes",
                "criteria": [
                    "Source nodes with risk level > 60%",
                    "Target nodes with systemic importance > 0.8",
                    "Path length ≤ 4 hops",
                    "Ranked by risk propagation potential"
                ]
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing critical paths: {str(e)}"
        )


@router.get("/vulnerability-assessment")
async def get_vulnerability_assessment():
    """
    Get comprehensive network vulnerability assessment.
    
    Returns detailed analysis of network resilience and risk factors.
    """
    try:
        analyzer = RiskNetworkAnalyzer()
        analysis = analyzer.analyze_network()
        
        # Category-wise risk analysis
        category_risks = {}
        category_counts = {}
        
        for node in analysis.nodes:
            cat = node.category
            if cat not in category_risks:
                category_risks[cat] = 0
                category_counts[cat] = 0
            category_risks[cat] += node.risk_level
            category_counts[cat] += 1
        
        category_analysis = {
            cat: {
                "avgRiskLevel": round(category_risks[cat] / category_counts[cat], 2),
                "nodeCount": category_counts[cat],
                "totalRisk": round(category_risks[cat], 2)
            }
            for cat in category_risks.keys()
        }
        
        # Risk distribution
        risk_distribution = {
            "low": len([n for n in analysis.nodes if n.risk_level < 30]),
            "medium": len([n for n in analysis.nodes if 30 <= n.risk_level < 60]),
            "high": len([n for n in analysis.nodes if 60 <= n.risk_level < 80]),
            "critical": len([n for n in analysis.nodes if n.risk_level >= 80])
        }
        
        return {
            "vulnerabilityScore": round(analysis.vulnerability_score, 2),
            "systemicRiskScore": round(analysis.systemic_risk_score, 2),
            "categoryAnalysis": category_analysis,
            "riskDistribution": risk_distribution,
            "networkMetrics": {
                "totalNodes": len(analysis.nodes),
                "totalConnections": len(analysis.edges),
                "avgRiskLevel": round(sum(n.risk_level for n in analysis.nodes) / len(analysis.nodes), 2),
                "highRiskNodes": len([n for n in analysis.nodes if n.risk_level > 60])
            },
            "recommendations": [
                "Monitor high-risk nodes closely" if analysis.vulnerability_score > 50 else "Maintain current monitoring",
                "Strengthen regulatory oversight" if analysis.systemic_risk_score > 60 else "Current oversight adequate",
                "Diversify supply chain dependencies" if any(n.category == 'supply_chain' and n.risk_level > 65 for n in analysis.nodes) else "Supply chain risks manageable"
            ],
            "analysisTimestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing vulnerability assessment: {str(e)}"
        )