"""
Supply Chain Disruption Risk Model
Advanced ML model for predicting supply chain disruption risks
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import networkx as nx
import joblib
import os

logger = logging.getLogger(__name__)


@dataclass
class SupplyChainPrediction:
    """Supply chain disruption prediction result"""
    overall_risk_score: float
    disruption_probability: float
    confidence: float
    risk_level: str
    critical_nodes: List[Dict[str, Any]]
    risk_factors: List[Dict[str, Any]]
    mitigation_strategies: List[str]
    time_horizon: str
    prediction_date: str
    model_version: str


@dataclass
class SupplyChainNode:
    """Supply chain network node"""
    node_id: str
    node_type: str  # supplier, manufacturer, distributor, retailer
    geographic_region: str
    capacity: float
    current_utilization: float
    criticality_score: float
    redundancy_level: float
    lead_time_days: int
    quality_score: float
    financial_stability: float


@dataclass
class SupplyChainMetrics:
    """Supply chain performance and risk metrics"""
    # Network topology metrics
    network_density: Optional[float] = None
    average_path_length: Optional[float] = None
    clustering_coefficient: Optional[float] = None
    
    # Performance metrics
    on_time_delivery_rate: Optional[float] = None
    inventory_turnover: Optional[float] = None
    capacity_utilization: Optional[float] = None
    lead_time_variability: Optional[float] = None
    
    # Risk indicators
    supplier_concentration: Optional[float] = None  # Herfindahl index
    geographic_concentration: Optional[float] = None
    single_source_dependencies: Optional[int] = None
    
    # External risk factors
    geopolitical_risk_score: Optional[float] = None
    weather_risk_score: Optional[float] = None
    economic_risk_score: Optional[float] = None
    cyber_risk_score: Optional[float] = None
    
    # Transportation metrics
    port_congestion_index: Optional[float] = None
    rail_capacity_utilization: Optional[float] = None
    trucking_capacity_utilization: Optional[float] = None
    fuel_price_volatility: Optional[float] = None


class SupplyChainRiskModel:
    """
    Advanced ML model for supply chain disruption risk assessment
    Uses network analysis and ensemble methods
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.models = {}
        self.scalers = {}
        self.network_analyzer = None
        self.anomaly_detector = None
        self.clustering_model = None
        self.model_path = model_path or "models/supply_chain_risk"
        self.version = "1.0.0"
        
        # Risk factor weights based on supply chain theory
        self.risk_weights = {
            "supplier_concentration": 0.15,
            "geographic_concentration": 0.12,
            "single_source_dependencies": 0.18,
            "capacity_utilization": 0.10,
            "lead_time_variability": 0.08,
            "geopolitical_risk": 0.12,
            "weather_risk": 0.08,
            "cyber_risk": 0.10,
            "financial_stability": 0.07
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ensemble of models for supply chain risk prediction"""
        
        # Model 1: Random Forest for overall risk scoring
        self.models['risk_scorer'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Model 2: Isolation Forest for anomaly detection
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        # Model 3: K-Means for node clustering and risk segmentation
        self.clustering_model = KMeans(
            n_clusters=5,
            random_state=42,
            n_init=10
        )
        
        # Scalers
        self.scalers['risk_scorer'] = StandardScaler()
        self.scalers['anomaly'] = MinMaxScaler()
        self.scalers['clustering'] = StandardScaler()
    
    def _create_supply_chain_network(self, nodes: List[SupplyChainNode]) -> nx.DiGraph:
        """Create supply chain network graph from nodes"""
        
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in nodes:
            G.add_node(
                node.node_id,
                node_type=node.node_type,
                region=node.geographic_region,
                capacity=node.capacity,
                utilization=node.current_utilization,
                criticality=node.criticality_score,
                redundancy=node.redundancy_level,
                lead_time=node.lead_time_days,
                quality=node.quality_score,
                financial=node.financial_stability
            )
        
        # Add edges based on typical supply chain flows
        # This would be based on actual supply chain relationships in production
        suppliers = [n for n in nodes if n.node_type == "supplier"]
        manufacturers = [n for n in nodes if n.node_type == "manufacturer"]
        distributors = [n for n in nodes if n.node_type == "distributor"]
        retailers = [n for n in nodes if n.node_type == "retailer"]
        
        # Create typical supply chain connections
        for supplier in suppliers:
            for manufacturer in manufacturers[:2]:  # Each supplier connects to 2 manufacturers
                G.add_edge(supplier.node_id, manufacturer.node_id, 
                          weight=1.0, flow_type="materials")
        
        for manufacturer in manufacturers:
            for distributor in distributors[:3]:  # Each manufacturer to 3 distributors
                G.add_edge(manufacturer.node_id, distributor.node_id,
                          weight=1.0, flow_type="products")
        
        for distributor in distributors:
            for retailer in retailers[:5]:  # Each distributor to 5 retailers
                G.add_edge(distributor.node_id, retailer.node_id,
                          weight=1.0, flow_type="distribution")
        
        return G
    
    def _calculate_network_metrics(self, network: nx.DiGraph) -> Dict[str, float]:
        """Calculate network topology metrics"""
        
        if len(network.nodes()) == 0:
            return {}
        
        try:
            # Basic network metrics
            density = nx.density(network)
            
            # For directed graphs, convert to undirected for some metrics
            undirected = network.to_undirected()
            
            # Average path length (for connected components)
            if nx.is_connected(undirected):
                avg_path_length = nx.average_shortest_path_length(undirected)
            else:
                # Calculate for largest connected component
                largest_cc = max(nx.connected_components(undirected), key=len)
                subgraph = undirected.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph)
            
            # Clustering coefficient
            clustering_coeff = nx.average_clustering(undirected)
            
            # Centrality measures
            betweenness = nx.betweenness_centrality(network)
            closeness = nx.closeness_centrality(network)
            
            return {
                "density": density,
                "avg_path_length": avg_path_length,
                "clustering_coefficient": clustering_coeff,
                "max_betweenness": max(betweenness.values()) if betweenness else 0,
                "avg_betweenness": np.mean(list(betweenness.values())) if betweenness else 0,
                "max_closeness": max(closeness.values()) if closeness else 0,
                "avg_closeness": np.mean(list(closeness.values())) if closeness else 0
            }
            
        except Exception as e:
            logger.warning(f"Error calculating network metrics: {e}")
            return {"density": 0, "avg_path_length": 0, "clustering_coefficient": 0}
    
    async def _generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data using real supply chain and infrastructure data"""
        from src.data.sources.cisa import CISAClient
        from src.data.sources.supply_chain import SupplyChainClient
        from src.core.config import get_settings
        
        settings = get_settings()
        
        try:
            # Fetch real supply chain data
            supply_chain_client = SupplyChainClient()
            cisa_client = CISAClient()
            
            # Get supply chain disruption data
            supply_chain_data = await supply_chain_client.get_disruption_data()
            infrastructure_data = await cisa_client.get_infrastructure_data()
            
            # Process real data into features
            features = []
            risk_scores = []
            
            # Convert real data to feature vectors
            for record in supply_chain_data.get('records', []):
                # Extract real supply chain metrics
                supplier_conc = record.get('supplier_concentration', 0.0)
                geo_conc = record.get('geographic_concentration', 0.0)
                single_source = record.get('single_source_dependencies', 0)
                capacity_util = record.get('capacity_utilization', 0.0)
                lead_time_var = record.get('lead_time_variability', 0.0)
                
                # External risk factors from real data
                geopolitical = record.get('geopolitical_risk', 0.0)
                weather = record.get('weather_risk', 0.0)
                cyber = record.get('cyber_risk', 0.0)
                financial = record.get('financial_stability', 0.0)
                
                # Network metrics from real infrastructure data
                network_density = record.get('network_density', 0.0)
                avg_path_length = record.get('average_path_length', 0.0)
                clustering = record.get('clustering_coefficient', 0.0)
                
                sample_features = [
                    supplier_conc, geo_conc, single_source, capacity_util,
                    lead_time_var, geopolitical, weather, cyber, financial,
                    network_density, avg_path_length, clustering
                ]
                
                # Use real risk score if available, otherwise calculate
                risk_score = record.get('overall_risk_score', self._calculate_risk_score(sample_features))
                
                features.append(sample_features)
                risk_scores.append(risk_score)
            
            if len(features) == 0:
                raise ValueError("No supply chain data available from external sources")
            
            logger.info(f"Created training dataset with {len(features)} real supply chain samples")
            return np.array(features), np.array(risk_scores)
            
        except Exception as e:
            logger.error(f"Failed to fetch real supply chain data: {str(e)}")
            logger.error("Cannot use synthetic data - real data required for production")
            raise ValueError("Real supply chain data is required for model training")
    
    def _calculate_risk_score(self, features: List[float]) -> float:
        """Calculate risk score from features"""
        supplier_conc, geo_conc, single_source, capacity_util, lead_time_var, \
        geopolitical, weather, cyber, financial, network_density, avg_path_length, clustering = features
        
        # Calculate risk score based on weighted factors
        risk_score = (
            supplier_conc * 0.15 +
            geo_conc * 0.12 +
            (single_source / 10) * 0.18 +  # Normalize single source count
            capacity_util * 0.10 +
            lead_time_var * 0.08 +
            geopolitical * 0.12 +
            weather * 0.08 +
            cyber * 0.10 +
            (1 - financial) * 0.07  # Higher financial instability = higher risk
        )
        
        # Add interaction effects
        interaction_effect = supplier_conc * geo_conc * 0.1
        risk_score = min(1.0, max(0.0, risk_score + interaction_effect))
        
        return risk_score
    
    async def train_models(self) -> Dict[str, float]:
        """Train all models in the ensemble"""
        
        # Get training data from real supply chain sources
        X, y = await self._generate_training_data()
        
        # Feature names
        feature_names = [
            'supplier_concentration', 'geographic_concentration', 'single_source_deps',
            'capacity_utilization', 'lead_time_variability', 'geopolitical_risk',
            'weather_risk', 'cyber_risk', 'financial_stability', 'network_density',
            'avg_path_length', 'clustering_coefficient'
        ]
        
        results = {}
        
        # Train risk scoring model
        logger.info("Training supply chain risk scoring model...")
        scaler = self.scalers['risk_scorer']
        X_scaled = scaler.fit_transform(X)
        
        # Time series cross-validation for risk scorer
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(
            self.models['risk_scorer'], X_scaled, y, 
            cv=tscv, scoring='neg_mean_squared_error'
        )
        
        # Train final model
        self.models['risk_scorer'].fit(X_scaled, y)
        
        results['risk_scorer'] = {
            'cv_rmse': np.sqrt(-cv_scores.mean()),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(feature_names, 
                                         self.models['risk_scorer'].feature_importances_))
        }
        
        # Train anomaly detection model
        logger.info("Training anomaly detection model...")
        anomaly_scaler = self.scalers['anomaly']
        X_anomaly = anomaly_scaler.fit_transform(X)
        self.anomaly_detector.fit(X_anomaly)
        
        results['anomaly_detector'] = {
            'contamination_rate': 0.1,
            'n_estimators': 100
        }
        
        # Train clustering model
        logger.info("Training clustering model...")
        cluster_scaler = self.scalers['clustering']
        X_cluster = cluster_scaler.fit_transform(X)
        self.clustering_model.fit(X_cluster)
        
        results['clustering'] = {
            'n_clusters': self.clustering_model.n_clusters,
            'inertia': self.clustering_model.inertia_
        }
        
        logger.info("Supply chain risk models training completed")
        return results
    
    def _prepare_features(self, metrics: SupplyChainMetrics) -> np.ndarray:
        """Prepare features from supply chain metrics"""
        
        features = [
            metrics.supplier_concentration or 0.5,
            metrics.geographic_concentration or 0.5,
            metrics.single_source_dependencies or 0,
            metrics.capacity_utilization or 0.8,
            metrics.lead_time_variability or 0.2,
            metrics.geopolitical_risk_score or 0.3,
            metrics.weather_risk_score or 0.2,
            metrics.cyber_risk_score or 0.2,
            (1 - (metrics.economic_risk_score or 0.7)),  # Financial stability proxy
            metrics.network_density or 0.3,
            metrics.average_path_length or 3.0,
            metrics.clustering_coefficient or 0.4
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict_supply_chain_risk(
        self,
        metrics: SupplyChainMetrics,
        nodes: Optional[List[SupplyChainNode]] = None
    ) -> SupplyChainPrediction:
        """Predict supply chain disruption risk"""
        
        # Prepare features
        X = self._prepare_features(metrics)
        
        # Get risk score prediction
        scaler = self.scalers['risk_scorer']
        X_scaled = scaler.transform(X)
        risk_score = self.models['risk_scorer'].predict(X_scaled)[0]
        
        # Get anomaly score
        anomaly_scaler = self.scalers['anomaly']
        X_anomaly = anomaly_scaler.transform(X)
        anomaly_score = self.anomaly_detector.decision_function(X_anomaly)[0]
        is_anomaly = self.anomaly_detector.predict(X_anomaly)[0] == -1
        
        # Get cluster assignment
        cluster_scaler = self.scalers['clustering']
        X_cluster = cluster_scaler.transform(X)
        cluster = self.clustering_model.predict(X_cluster)[0]
        
        # Combine scores for overall risk
        overall_risk = risk_score * 100
        if is_anomaly:
            overall_risk = min(100, overall_risk * 1.2)  # Boost risk for anomalies
        
        # Calculate disruption probability
        disruption_prob = 1 / (1 + np.exp(-5 * (risk_score - 0.6)))  # Sigmoid transformation
        
        # Determine risk level
        if overall_risk >= 80:
            risk_level = "Critical"
        elif overall_risk >= 60:
            risk_level = "High"
        elif overall_risk >= 40:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Calculate confidence (based on model certainty)
        confidence = max(0.6, 1.0 - abs(anomaly_score) * 0.1)
        
        # Identify critical nodes and risk factors
        critical_nodes = self._identify_critical_nodes(nodes) if nodes else []
        risk_factors = self._identify_risk_factors(metrics, risk_score)
        mitigation_strategies = self._generate_mitigation_strategies(risk_factors)
        
        return SupplyChainPrediction(
            overall_risk_score=overall_risk,
            disruption_probability=disruption_prob,
            confidence=confidence,
            risk_level=risk_level,
            critical_nodes=critical_nodes,
            risk_factors=risk_factors,
            mitigation_strategies=mitigation_strategies,
            time_horizon="6_months",
            prediction_date=datetime.utcnow().isoformat(),
            model_version=self.version
        )
    
    def _identify_critical_nodes(self, nodes: List[SupplyChainNode]) -> List[Dict[str, Any]]:
        """Identify critical nodes in supply chain network"""
        
        critical_nodes = []
        
        # Create network and calculate centrality
        if len(nodes) > 0:
            network = self._create_supply_chain_network(nodes)
            betweenness = nx.betweenness_centrality(network)
            
            # Find nodes with high centrality or low redundancy
            for node in nodes:
                criticality_factors = []
                
                # High betweenness centrality
                if betweenness.get(node.node_id, 0) > 0.1:
                    criticality_factors.append("High network centrality")
                
                # Low redundancy
                if node.redundancy_level < 0.3:
                    criticality_factors.append("Low redundancy")
                
                # High utilization
                if node.current_utilization > 0.9:
                    criticality_factors.append("High capacity utilization")
                
                # Poor financial stability
                if node.financial_stability < 0.6:
                    criticality_factors.append("Financial instability")
                
                if criticality_factors:
                    critical_nodes.append({
                        "node_id": node.node_id,
                        "node_type": node.node_type,
                        "region": node.geographic_region,
                        "criticality_score": node.criticality_score,
                        "factors": criticality_factors,
                        "betweenness_centrality": betweenness.get(node.node_id, 0)
                    })
        
        # Sort by criticality score
        critical_nodes.sort(key=lambda x: x["criticality_score"], reverse=True)
        return critical_nodes[:10]  # Top 10 critical nodes
    
    def _identify_risk_factors(
        self, 
        metrics: SupplyChainMetrics, 
        risk_score: float
    ) -> List[Dict[str, Any]]:
        """Identify key risk factors contributing to supply chain risk"""
        
        risk_factors = []
        
        # Supplier concentration risk
        if metrics.supplier_concentration and metrics.supplier_concentration > 0.7:
            risk_factors.append({
                "factor": "High Supplier Concentration",
                "value": metrics.supplier_concentration,
                "impact": "high",
                "description": "Over-reliance on few suppliers increases disruption risk"
            })
        
        # Geographic concentration risk
        if metrics.geographic_concentration and metrics.geographic_concentration > 0.6:
            risk_factors.append({
                "factor": "Geographic Concentration",
                "value": metrics.geographic_concentration,
                "impact": "medium",
                "description": "Suppliers concentrated in few regions increases regional risk exposure"
            })
        
        # Single source dependencies
        if metrics.single_source_dependencies and metrics.single_source_dependencies > 5:
            risk_factors.append({
                "factor": "Single Source Dependencies",
                "value": metrics.single_source_dependencies,
                "impact": "high",
                "description": f"{metrics.single_source_dependencies} critical components have single source"
            })
        
        # Capacity utilization
        if metrics.capacity_utilization and metrics.capacity_utilization > 0.9:
            risk_factors.append({
                "factor": "High Capacity Utilization",
                "value": metrics.capacity_utilization,
                "impact": "medium",
                "description": "Limited spare capacity reduces flexibility during disruptions"
            })
        
        # External risk factors
        if metrics.geopolitical_risk_score and metrics.geopolitical_risk_score > 0.6:
            risk_factors.append({
                "factor": "Geopolitical Risk",
                "value": metrics.geopolitical_risk_score,
                "impact": "high",
                "description": "High geopolitical tensions in supplier regions"
            })
        
        if metrics.cyber_risk_score and metrics.cyber_risk_score > 0.5:
            risk_factors.append({
                "factor": "Cybersecurity Risk",
                "value": metrics.cyber_risk_score,
                "impact": "medium",
                "description": "Elevated cyber threat levels affecting supply chain digitization"
            })
        
        return risk_factors
    
    def _generate_mitigation_strategies(self, risk_factors: List[Dict[str, Any]]) -> List[str]:
        """Generate mitigation strategies based on identified risks"""
        
        strategies = []
        
        for factor in risk_factors:
            factor_name = factor["factor"]
            
            if "Supplier Concentration" in factor_name:
                strategies.append("Diversify supplier base across multiple vendors and regions")
            
            elif "Geographic Concentration" in factor_name:
                strategies.append("Establish suppliers in geographically diverse locations")
            
            elif "Single Source Dependencies" in factor_name:
                strategies.append("Identify and qualify alternative suppliers for critical components")
            
            elif "Capacity Utilization" in factor_name:
                strategies.append("Increase capacity buffers and flexible manufacturing capabilities")
            
            elif "Geopolitical Risk" in factor_name:
                strategies.append("Develop contingency plans for geopolitically sensitive regions")
            
            elif "Cybersecurity Risk" in factor_name:
                strategies.append("Enhance cybersecurity measures across supply chain partners")
        
        # Add general strategies
        strategies.extend([
            "Implement real-time supply chain visibility and monitoring",
            "Develop supplier financial health monitoring programs",
            "Create strategic inventory buffers for critical components",
            "Establish rapid supplier switching capabilities"
        ])
        
        return list(set(strategies))  # Remove duplicates
    
    def save_models(self, path: Optional[str] = None):
        """Save trained models to disk"""
        save_path = path or self.model_path
        os.makedirs(save_path, exist_ok=True)
        
        # Save all models and scalers
        joblib.dump(self.models['risk_scorer'], f"{save_path}/risk_scorer.pkl")
        joblib.dump(self.anomaly_detector, f"{save_path}/anomaly_detector.pkl")
        joblib.dump(self.clustering_model, f"{save_path}/clustering_model.pkl")
        
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{save_path}/{name}_scaler.pkl")
        
        # Save metadata
        metadata = {
            "version": self.version,
            "risk_weights": self.risk_weights
        }
        joblib.dump(metadata, f"{save_path}/metadata.pkl")
        
        logger.info(f"Supply chain models saved to {save_path}")
    
    def load_models(self, path: Optional[str] = None):
        """Load trained models from disk"""
        load_path = path or self.model_path
        
        try:
            # Load models
            self.models['risk_scorer'] = joblib.load(f"{load_path}/risk_scorer.pkl")
            self.anomaly_detector = joblib.load(f"{load_path}/anomaly_detector.pkl")
            self.clustering_model = joblib.load(f"{load_path}/clustering_model.pkl")
            
            # Load scalers
            for name in self.scalers.keys():
                self.scalers[name] = joblib.load(f"{load_path}/{name}_scaler.pkl")
            
            # Load metadata
            metadata = joblib.load(f"{load_path}/metadata.pkl")
            self.version = metadata.get("version", self.version)
            self.risk_weights = metadata.get("risk_weights", self.risk_weights)
            
            logger.info(f"Supply chain models loaded from {load_path}")
            
        except FileNotFoundError:
            logger.warning(f"No saved models found at {load_path}, using default models")


# Convenience functions for integration
async def predict_supply_chain_disruption(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Predict supply chain disruption risk from metrics dictionary"""
    
    # Convert dictionary to SupplyChainMetrics object
    supply_metrics = SupplyChainMetrics(
        supplier_concentration=metrics.get("supplier_concentration"),
        geographic_concentration=metrics.get("geographic_concentration"),
        single_source_dependencies=metrics.get("single_source_dependencies"),
        capacity_utilization=metrics.get("capacity_utilization"),
        lead_time_variability=metrics.get("lead_time_variability"),
        geopolitical_risk_score=metrics.get("geopolitical_risk"),
        weather_risk_score=metrics.get("weather_risk"),
        economic_risk_score=metrics.get("economic_risk"),
        cyber_risk_score=metrics.get("cyber_risk"),
        network_density=metrics.get("network_density"),
        average_path_length=metrics.get("average_path_length"),
        clustering_coefficient=metrics.get("clustering_coefficient")
    )
    
    # Create model and get prediction
    model = SupplyChainRiskModel()
    
    # Train models if not already trained
    try:
        model.load_models()
    except:
        logger.info("Training supply chain risk models...")
        model.train_models()
        model.save_models()
    
    prediction = model.predict_supply_chain_risk(supply_metrics)
    
    return {
        "overall_risk_score": prediction.overall_risk_score,
        "disruption_probability": prediction.disruption_probability,
        "confidence": prediction.confidence,
        "risk_level": prediction.risk_level,
        "critical_nodes": prediction.critical_nodes,
        "risk_factors": prediction.risk_factors,
        "mitigation_strategies": prediction.mitigation_strategies,
        "time_horizon": prediction.time_horizon,
        "prediction_date": prediction.prediction_date,
        "model_version": prediction.model_version
    }


async def train_supply_chain_models() -> Dict[str, Any]:
    """Train supply chain risk models using real data"""
    
    model = SupplyChainRiskModel()
    training_results = await model.train_models()
    model.save_models()
    
    return {
        "status": "completed",
        "training_results": training_results,
        "model_version": model.version,
        "timestamp": datetime.utcnow().isoformat()
    }