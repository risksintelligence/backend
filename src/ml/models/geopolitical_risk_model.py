"""
Geopolitical Risk Assessment Model
Advanced ML model for analyzing and predicting geopolitical risks
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

logger = logging.getLogger(__name__)


@dataclass
class GeopoliticalRiskPrediction:
    """Geopolitical risk assessment result"""
    overall_risk_score: float
    risk_level: str  # "Low", "Medium", "High", "Critical"
    risk_category: str  # "Political", "Economic", "Military", "Diplomatic", "Hybrid"
    probability_escalation: float
    confidence: float
    active_tensions: List[Dict[str, Any]]
    risk_factors: List[Dict[str, Any]]
    affected_regions: List[str]
    economic_impact_estimate: float
    mitigation_recommendations: List[str]
    time_horizon: str
    prediction_date: str
    model_version: str


@dataclass
class GeopoliticalIndicators:
    """Geopolitical indicators for risk modeling"""
    # Political stability indicators
    political_stability_index: Optional[float] = None
    government_effectiveness: Optional[float] = None
    regulatory_quality: Optional[float] = None
    rule_of_law: Optional[float] = None
    
    # Conflict indicators
    active_conflicts_count: Optional[int] = None
    conflict_intensity_index: Optional[float] = None
    military_expenditure_gdp: Optional[float] = None
    arms_imports_volume: Optional[float] = None
    
    # Economic pressure indicators
    sanctions_index: Optional[float] = None
    trade_restrictions_count: Optional[int] = None
    currency_volatility: Optional[float] = None
    inflation_rate: Optional[float] = None
    
    # Diplomatic indicators
    diplomatic_relations_index: Optional[float] = None
    international_treaties_count: Optional[int] = None
    un_security_council_votes: Optional[float] = None  # Alignment score
    
    # Social indicators
    social_unrest_index: Optional[float] = None
    press_freedom_index: Optional[float] = None
    corruption_perception_index: Optional[float] = None
    human_development_index: Optional[float] = None
    
    # Security indicators
    terrorism_index: Optional[float] = None
    cyber_attacks_count: Optional[int] = None
    border_security_index: Optional[float] = None
    
    # Economic interdependence
    trade_dependency_ratio: Optional[float] = None
    fdi_volatility: Optional[float] = None
    supply_chain_exposure: Optional[float] = None


class GeopoliticalRiskModel:
    """
    Advanced ML model for geopolitical risk assessment
    Uses multiple classifiers and risk scenario analysis
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.risk_thresholds = {}
        self.model_path = model_path or "models/geopolitical_risk"
        self.version = "1.0.0"
        
        # Risk level thresholds
        self.risk_thresholds = {
            "low": 0.25,
            "medium": 0.50,
            "high": 0.75,
            "critical": 1.0
        }
        
        # Feature weights based on geopolitical research
        self.feature_weights = {
            "political_stability": 0.18,
            "active_conflicts": 0.16,
            "sanctions_index": 0.12,
            "military_expenditure": 0.10,
            "social_unrest": 0.10,
            "diplomatic_relations": 0.08,
            "terrorism_index": 0.08,
            "economic_pressure": 0.08,
            "corruption_index": 0.06,
            "cyber_attacks": 0.04
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ensemble of models for geopolitical risk assessment"""
        
        # Model 1: Random Forest (interpretable feature importance)
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        # Model 2: Gradient Boosting (sequential learning)
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=5,
            random_state=42
        )
        
        # Model 3: Neural Network (complex pattern recognition)
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # Scalers for each model
        self.scalers['random_forest'] = StandardScaler()
        self.scalers['gradient_boosting'] = StandardScaler()
        self.scalers['neural_network'] = StandardScaler()
        
        # Label encoder for risk categories
        self.label_encoders['risk_category'] = LabelEncoder()
    
    def _generate_geopolitical_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic training data based on geopolitical risk patterns
        In production, this would use real geopolitical data
        """
        np.random.seed(42)
        n_samples = 1500
        
        features = []
        risk_levels = []
        risk_categories = []
        
        # Define risk scenarios
        scenarios = [
            'stable_democracy', 'emerging_tensions', 'active_conflict', 
            'economic_warfare', 'diplomatic_crisis', 'hybrid_threats'
        ]
        
        for i in range(n_samples):
            # Choose scenario
            scenario = np.random.choice(scenarios, p=[0.25, 0.25, 0.15, 0.15, 0.10, 0.10])
            
            if scenario == 'stable_democracy':
                # Stable democratic country
                political_stability = np.random.normal(0.8, 0.1)
                government_effectiveness = np.random.normal(0.8, 0.1)
                active_conflicts = np.random.poisson(0.2)
                conflict_intensity = np.random.beta(1, 9)
                sanctions_index = np.random.beta(1, 9)
                social_unrest = np.random.beta(2, 8)
                terrorism_index = np.random.beta(1, 9)
                risk_level = 0  # Low risk
                risk_category = 'Political'
                
            elif scenario == 'emerging_tensions':
                # Emerging diplomatic tensions
                political_stability = np.random.normal(0.6, 0.15)
                government_effectiveness = np.random.normal(0.6, 0.15)
                active_conflicts = np.random.poisson(1)
                conflict_intensity = np.random.beta(3, 7)
                sanctions_index = np.random.beta(2, 6)
                social_unrest = np.random.beta(3, 5)
                terrorism_index = np.random.beta(2, 6)
                risk_level = 1  # Medium risk
                risk_category = 'Diplomatic'
                
            elif scenario == 'active_conflict':
                # Active military conflict
                political_stability = np.random.normal(0.3, 0.15)
                government_effectiveness = np.random.normal(0.4, 0.15)
                active_conflicts = np.random.poisson(3)
                conflict_intensity = np.random.beta(7, 3)
                sanctions_index = np.random.beta(5, 3)
                social_unrest = np.random.beta(6, 3)
                terrorism_index = np.random.beta(4, 4)
                risk_level = 3  # Critical risk
                risk_category = 'Military'
                
            elif scenario == 'economic_warfare':
                # Economic sanctions and trade wars
                political_stability = np.random.normal(0.5, 0.15)
                government_effectiveness = np.random.normal(0.5, 0.15)
                active_conflicts = np.random.poisson(1)
                conflict_intensity = np.random.beta(2, 6)
                sanctions_index = np.random.beta(7, 2)
                social_unrest = np.random.beta(4, 4)
                terrorism_index = np.random.beta(2, 6)
                risk_level = 2  # High risk
                risk_category = 'Economic'
                
            elif scenario == 'diplomatic_crisis':
                # Diplomatic breakdown
                political_stability = np.random.normal(0.4, 0.15)
                government_effectiveness = np.random.normal(0.5, 0.15)
                active_conflicts = np.random.poisson(1)
                conflict_intensity = np.random.beta(3, 5)
                sanctions_index = np.random.beta(4, 4)
                social_unrest = np.random.beta(5, 3)
                terrorism_index = np.random.beta(3, 5)
                risk_level = 2  # High risk
                risk_category = 'Diplomatic'
                
            else:  # hybrid_threats
                # Hybrid warfare (cyber, information, etc.)
                political_stability = np.random.normal(0.5, 0.15)
                government_effectiveness = np.random.normal(0.6, 0.15)
                active_conflicts = np.random.poisson(1)
                conflict_intensity = np.random.beta(2, 6)
                sanctions_index = np.random.beta(3, 5)
                social_unrest = np.random.beta(4, 4)
                terrorism_index = np.random.beta(5, 3)
                risk_level = 2  # High risk
                risk_category = 'Hybrid'
            
            # Additional features
            military_expenditure = np.random.normal(0.025, 0.015) + (0.02 if scenario == 'active_conflict' else 0)
            trade_restrictions = np.random.poisson(2) + (5 if scenario == 'economic_warfare' else 0)
            diplomatic_relations = 1 - sanctions_index + np.random.normal(0, 0.1)
            corruption_index = 1 - government_effectiveness + np.random.normal(0, 0.1)
            press_freedom = political_stability + np.random.normal(0, 0.1)
            cyber_attacks = np.random.poisson(2) + (10 if scenario == 'hybrid_threats' else 0)
            
            # Economic indicators
            currency_volatility = 0.1 + (sanctions_index * 0.3) + np.random.normal(0, 0.05)
            inflation_rate = np.random.normal(0.03, 0.02) + (sanctions_index * 0.05)
            trade_dependency = np.random.beta(3, 3)
            
            # Clamp values to realistic ranges
            political_stability = max(0, min(1, political_stability))
            government_effectiveness = max(0, min(1, government_effectiveness))
            sanctions_index = max(0, min(1, sanctions_index))
            social_unrest = max(0, min(1, social_unrest))
            terrorism_index = max(0, min(1, terrorism_index))
            
            sample_features = [
                political_stability, government_effectiveness, active_conflicts,
                conflict_intensity, military_expenditure, sanctions_index,
                trade_restrictions, diplomatic_relations, social_unrest,
                terrorism_index, corruption_index, press_freedom,
                cyber_attacks, currency_volatility, inflation_rate,
                trade_dependency
            ]
            
            features.append(sample_features)
            risk_levels.append(risk_level)
            risk_categories.append(risk_category)
        
        return np.array(features), np.array(risk_levels), np.array(risk_categories)
    
    def train_models(self) -> Dict[str, Any]:
        """Train all models in the ensemble"""
        
        # Get training data
        X, y_risk, y_category = self._generate_geopolitical_training_data()
        
        # Feature names
        feature_names = [
            'political_stability', 'government_effectiveness', 'active_conflicts',
            'conflict_intensity', 'military_expenditure', 'sanctions_index',
            'trade_restrictions', 'diplomatic_relations', 'social_unrest',
            'terrorism_index', 'corruption_index', 'press_freedom',
            'cyber_attacks', 'currency_volatility', 'inflation_rate',
            'trade_dependency'
        ]
        
        # Encode categorical labels
        self.label_encoders['risk_category'].fit(y_category)
        y_category_encoded = self.label_encoders['risk_category'].transform(y_category)
        
        results = {}
        
        # Train each model for risk level prediction
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} geopolitical risk model...")
            
            # Scale features
            scaler = self.scalers[model_name]
            X_scaled = scaler.fit_transform(X)
            
            # Stratified cross-validation for risk level
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_scaled, y_risk, cv=skf, scoring='accuracy')
            
            # Train final model
            model.fit(X_scaled, y_risk)
            
            # Predictions for evaluation
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names, model.feature_importances_))
                self.feature_importance[model_name] = importance
            
            # Calculate metrics
            if len(np.unique(y_risk)) > 2:  # Multi-class
                auc_score = roc_auc_score(y_risk, y_pred_proba, multi_class='ovr', average='weighted')
            else:
                auc_score = roc_auc_score(y_risk, y_pred_proba[:, 1])
            
            results[model_name] = {
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_accuracy': (y_pred == y_risk).mean(),
                'auc_score': auc_score,
                'classification_report': classification_report(y_risk, y_pred, output_dict=True)
            }
            
            logger.info(f"{model_name} - CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return results
    
    def _prepare_features(self, indicators: GeopoliticalIndicators) -> np.ndarray:
        """Prepare features from geopolitical indicators"""
        
        features = [
            indicators.political_stability_index or 0.5,
            indicators.government_effectiveness or 0.5,
            indicators.active_conflicts_count or 0,
            indicators.conflict_intensity_index or 0.0,
            indicators.military_expenditure_gdp or 0.02,
            indicators.sanctions_index or 0.0,
            indicators.trade_restrictions_count or 0,
            indicators.diplomatic_relations_index or 0.5,
            indicators.social_unrest_index or 0.2,
            indicators.terrorism_index or 0.1,
            (1 - (indicators.corruption_perception_index or 0.5)),  # Corruption level
            indicators.press_freedom_index or 0.5,
            indicators.cyber_attacks_count or 0,
            indicators.currency_volatility or 0.1,
            indicators.inflation_rate or 0.03,
            indicators.trade_dependency_ratio or 0.3
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict_geopolitical_risk(
        self,
        indicators: GeopoliticalIndicators,
        region: str = "Global"
    ) -> GeopoliticalRiskPrediction:
        """Predict geopolitical risk using ensemble of models"""
        
        # Prepare features
        X = self._prepare_features(indicators)
        
        # Get predictions from all models
        risk_predictions = {}
        risk_probabilities = {}
        
        for model_name, model in self.models.items():
            # Scale features
            scaler = self.scalers[model_name]
            X_scaled = scaler.transform(X)
            
            # Get risk level prediction
            risk_level = model.predict(X_scaled)[0]
            risk_proba = model.predict_proba(X_scaled)[0]
            
            risk_predictions[model_name] = risk_level
            risk_probabilities[model_name] = risk_proba
        
        # Ensemble prediction (majority vote with confidence weighting)
        ensemble_risk_level = int(np.round(np.mean(list(risk_predictions.values()))))
        
        # Calculate average probabilities
        avg_probabilities = np.mean([proba for proba in risk_probabilities.values()], axis=0)
        overall_risk_score = np.sum(avg_probabilities * np.array([0.1, 0.3, 0.6, 1.0]))  # Weighted by severity
        
        # Determine risk level string
        risk_level_map = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
        risk_level_str = risk_level_map.get(ensemble_risk_level, "Medium")
        
        # Calculate escalation probability
        escalation_prob = avg_probabilities[2] + avg_probabilities[3]  # High + Critical probabilities
        
        # Calculate confidence (based on model agreement)
        pred_std = np.std(list(risk_predictions.values()))
        confidence = max(0.6, 1.0 - (pred_std * 0.3))
        
        # Determine risk category (simplified - in production would use separate model)
        risk_category = self._determine_risk_category(indicators)
        
        # Identify active tensions and risk factors
        active_tensions = self._identify_active_tensions(indicators)
        risk_factors = self._identify_risk_factors(indicators, overall_risk_score)
        
        # Estimate economic impact
        economic_impact = self._estimate_economic_impact(overall_risk_score, indicators)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_factors, risk_level_str)
        
        return GeopoliticalRiskPrediction(
            overall_risk_score=overall_risk_score * 100,
            risk_level=risk_level_str,
            risk_category=risk_category,
            probability_escalation=escalation_prob,
            confidence=confidence,
            active_tensions=active_tensions,
            risk_factors=risk_factors,
            affected_regions=[region],
            economic_impact_estimate=economic_impact,
            mitigation_recommendations=recommendations,
            time_horizon="6_months",
            prediction_date=datetime.utcnow().isoformat(),
            model_version=self.version
        )
    
    def _determine_risk_category(self, indicators: GeopoliticalIndicators) -> str:
        """Determine primary risk category based on indicators"""
        
        category_scores = {
            "Political": 0,
            "Economic": 0,
            "Military": 0,
            "Diplomatic": 0,
            "Hybrid": 0
        }
        
        # Political instability
        if indicators.political_stability_index and indicators.political_stability_index < 0.5:
            category_scores["Political"] += 3
        if indicators.social_unrest_index and indicators.social_unrest_index > 0.5:
            category_scores["Political"] += 2
        
        # Economic pressures
        if indicators.sanctions_index and indicators.sanctions_index > 0.3:
            category_scores["Economic"] += 3
        if indicators.trade_restrictions_count and indicators.trade_restrictions_count > 5:
            category_scores["Economic"] += 2
        
        # Military factors
        if indicators.active_conflicts_count and indicators.active_conflicts_count > 1:
            category_scores["Military"] += 3
        if indicators.military_expenditure_gdp and indicators.military_expenditure_gdp > 0.04:
            category_scores["Military"] += 2
        
        # Diplomatic issues
        if indicators.diplomatic_relations_index and indicators.diplomatic_relations_index < 0.4:
            category_scores["Diplomatic"] += 3
        
        # Hybrid threats
        if indicators.cyber_attacks_count and indicators.cyber_attacks_count > 10:
            category_scores["Hybrid"] += 3
        if indicators.terrorism_index and indicators.terrorism_index > 0.5:
            category_scores["Hybrid"] += 2
        
        # Return category with highest score
        return max(category_scores, key=category_scores.get)
    
    def _identify_active_tensions(self, indicators: GeopoliticalIndicators) -> List[Dict[str, Any]]:
        """Identify current active tensions"""
        
        tensions = []
        
        # Armed conflicts
        if indicators.active_conflicts_count and indicators.active_conflicts_count > 0:
            tensions.append({
                "type": "Armed Conflict",
                "severity": "high" if indicators.active_conflicts_count > 2 else "medium",
                "description": f"{indicators.active_conflicts_count} active armed conflicts",
                "impact": "Direct military threat and regional instability"
            })
        
        # Economic sanctions
        if indicators.sanctions_index and indicators.sanctions_index > 0.3:
            tensions.append({
                "type": "Economic Sanctions",
                "severity": "high" if indicators.sanctions_index > 0.6 else "medium",
                "description": f"Sanctions index at {indicators.sanctions_index:.2f}",
                "impact": "Trade disruption and economic pressure"
            })
        
        # Social unrest
        if indicators.social_unrest_index and indicators.social_unrest_index > 0.4:
            tensions.append({
                "type": "Social Unrest",
                "severity": "high" if indicators.social_unrest_index > 0.7 else "medium",
                "description": f"Social unrest index at {indicators.social_unrest_index:.2f}",
                "impact": "Internal instability and governance challenges"
            })
        
        # Cyber warfare
        if indicators.cyber_attacks_count and indicators.cyber_attacks_count > 5:
            tensions.append({
                "type": "Cyber Warfare",
                "severity": "high" if indicators.cyber_attacks_count > 15 else "medium",
                "description": f"{indicators.cyber_attacks_count} cyber attacks recorded",
                "impact": "Critical infrastructure and information systems at risk"
            })
        
        return tensions
    
    def _identify_risk_factors(
        self, 
        indicators: GeopoliticalIndicators, 
        risk_score: float
    ) -> List[Dict[str, Any]]:
        """Identify key risk factors"""
        
        risk_factors = []
        
        # Political instability
        if indicators.political_stability_index and indicators.political_stability_index < 0.4:
            risk_factors.append({
                "factor": "Political Instability",
                "value": indicators.political_stability_index,
                "impact": "high",
                "description": "Low political stability threatens governance and decision-making"
            })
        
        # Weak institutions
        if indicators.government_effectiveness and indicators.government_effectiveness < 0.4:
            risk_factors.append({
                "factor": "Weak Government Institutions",
                "value": indicators.government_effectiveness,
                "impact": "medium",
                "description": "Ineffective government reduces crisis response capability"
            })
        
        # High military spending
        if indicators.military_expenditure_gdp and indicators.military_expenditure_gdp > 0.05:
            risk_factors.append({
                "factor": "High Military Expenditure",
                "value": indicators.military_expenditure_gdp,
                "impact": "medium",
                "description": f"Military spending at {indicators.military_expenditure_gdp:.1%} of GDP indicates tensions"
            })
        
        # Economic vulnerabilities
        if indicators.currency_volatility and indicators.currency_volatility > 0.2:
            risk_factors.append({
                "factor": "Currency Instability",
                "value": indicators.currency_volatility,
                "impact": "medium",
                "description": "High currency volatility indicates economic stress"
            })
        
        # Terrorism threat
        if indicators.terrorism_index and indicators.terrorism_index > 0.4:
            risk_factors.append({
                "factor": "Terrorism Threat",
                "value": indicators.terrorism_index,
                "impact": "high",
                "description": "Elevated terrorism index indicates security threats"
            })
        
        return risk_factors
    
    def _estimate_economic_impact(
        self, 
        risk_score: float, 
        indicators: GeopoliticalIndicators
    ) -> float:
        """Estimate potential economic impact of geopolitical risks"""
        
        base_impact = risk_score * 0.05  # Base 5% impact for maximum risk
        
        # Adjust based on economic exposure
        if indicators.trade_dependency_ratio:
            base_impact *= (1 + indicators.trade_dependency_ratio)
        
        # Adjust for sanctions
        if indicators.sanctions_index:
            base_impact *= (1 + indicators.sanctions_index * 0.5)
        
        # Adjust for currency volatility
        if indicators.currency_volatility:
            base_impact *= (1 + indicators.currency_volatility)
        
        return min(0.25, base_impact)  # Cap at 25% impact
    
    def _generate_recommendations(
        self, 
        risk_factors: List[Dict[str, Any]], 
        risk_level: str
    ) -> List[str]:
        """Generate risk mitigation recommendations"""
        
        recommendations = []
        
        # General recommendations by risk level
        if risk_level in ["High", "Critical"]:
            recommendations.extend([
                "Implement enhanced security protocols for critical assets",
                "Develop contingency plans for supply chain disruptions",
                "Increase diplomatic engagement to reduce tensions"
            ])
        
        # Specific recommendations based on risk factors
        for factor in risk_factors:
            factor_name = factor["factor"]
            
            if "Political Instability" in factor_name:
                recommendations.append("Monitor political developments and prepare for regime changes")
            
            elif "Military Expenditure" in factor_name:
                recommendations.append("Assess military build-up implications and prepare defensive measures")
            
            elif "Currency Instability" in factor_name:
                recommendations.append("Hedge currency exposure and diversify financial holdings")
            
            elif "Terrorism Threat" in factor_name:
                recommendations.append("Enhance counter-terrorism measures and intelligence sharing")
        
        # Add general risk management recommendations
        recommendations.extend([
            "Diversify geopolitical exposure across regions",
            "Strengthen international partnerships and alliances",
            "Invest in early warning systems for geopolitical events",
            "Develop rapid response capabilities for crisis situations"
        ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def save_models(self, path: Optional[str] = None):
        """Save trained models to disk"""
        save_path = path or self.model_path
        os.makedirs(save_path, exist_ok=True)
        
        # Save models, scalers, and encoders
        for model_name, model in self.models.items():
            joblib.dump(model, f"{save_path}/{model_name}_model.pkl")
            joblib.dump(self.scalers[model_name], f"{save_path}/{model_name}_scaler.pkl")
        
        for encoder_name, encoder in self.label_encoders.items():
            joblib.dump(encoder, f"{save_path}/{encoder_name}_encoder.pkl")
        
        # Save metadata
        metadata = {
            "version": self.version,
            "feature_importance": self.feature_importance,
            "feature_weights": self.feature_weights,
            "risk_thresholds": self.risk_thresholds
        }
        joblib.dump(metadata, f"{save_path}/metadata.pkl")
        
        logger.info(f"Geopolitical risk models saved to {save_path}")
    
    def load_models(self, path: Optional[str] = None):
        """Load trained models from disk"""
        load_path = path or self.model_path
        
        try:
            # Load models and scalers
            for model_name in self.models.keys():
                self.models[model_name] = joblib.load(f"{load_path}/{model_name}_model.pkl")
                self.scalers[model_name] = joblib.load(f"{load_path}/{model_name}_scaler.pkl")
            
            # Load encoders
            for encoder_name in self.label_encoders.keys():
                self.label_encoders[encoder_name] = joblib.load(f"{load_path}/{encoder_name}_encoder.pkl")
            
            # Load metadata
            metadata = joblib.load(f"{load_path}/metadata.pkl")
            self.version = metadata.get("version", self.version)
            self.feature_importance = metadata.get("feature_importance", {})
            self.feature_weights = metadata.get("feature_weights", self.feature_weights)
            self.risk_thresholds = metadata.get("risk_thresholds", self.risk_thresholds)
            
            logger.info(f"Geopolitical risk models loaded from {load_path}")
            
        except FileNotFoundError:
            logger.warning(f"No saved models found at {load_path}, using default models")


# Convenience functions for integration
async def predict_geopolitical_risk(indicators: Dict[str, float]) -> Dict[str, Any]:
    """Predict geopolitical risk from indicators dictionary"""
    
    # Convert dictionary to GeopoliticalIndicators object
    geopolitical_indicators = GeopoliticalIndicators(
        political_stability_index=indicators.get("political_stability"),
        government_effectiveness=indicators.get("government_effectiveness"),
        active_conflicts_count=indicators.get("active_conflicts"),
        conflict_intensity_index=indicators.get("conflict_intensity"),
        military_expenditure_gdp=indicators.get("military_expenditure"),
        sanctions_index=indicators.get("sanctions_index"),
        trade_restrictions_count=indicators.get("trade_restrictions"),
        diplomatic_relations_index=indicators.get("diplomatic_relations"),
        social_unrest_index=indicators.get("social_unrest"),
        terrorism_index=indicators.get("terrorism_index"),
        corruption_perception_index=indicators.get("corruption_index"),
        press_freedom_index=indicators.get("press_freedom"),
        cyber_attacks_count=indicators.get("cyber_attacks"),
        currency_volatility=indicators.get("currency_volatility"),
        inflation_rate=indicators.get("inflation_rate"),
        trade_dependency_ratio=indicators.get("trade_dependency")
    )
    
    # Create model and get prediction
    model = GeopoliticalRiskModel()
    
    # Train models if not already trained
    try:
        model.load_models()
    except:
        logger.info("Training geopolitical risk models...")
        model.train_models()
        model.save_models()
    
    prediction = model.predict_geopolitical_risk(geopolitical_indicators)
    
    return {
        "overall_risk_score": prediction.overall_risk_score,
        "risk_level": prediction.risk_level,
        "risk_category": prediction.risk_category,
        "probability_escalation": prediction.probability_escalation,
        "confidence": prediction.confidence,
        "active_tensions": prediction.active_tensions,
        "risk_factors": prediction.risk_factors,
        "affected_regions": prediction.affected_regions,
        "economic_impact_estimate": prediction.economic_impact_estimate,
        "mitigation_recommendations": prediction.mitigation_recommendations,
        "time_horizon": prediction.time_horizon,
        "prediction_date": prediction.prediction_date,
        "model_version": prediction.model_version
    }


async def train_geopolitical_models() -> Dict[str, Any]:
    """Train geopolitical risk models"""
    
    model = GeopoliticalRiskModel()
    training_results = model.train_models()
    model.save_models()
    
    return {
        "status": "completed",
        "training_results": training_results,
        "model_version": model.version,
        "models_trained": list(model.models.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }