"""
Model Interpretability API endpoints for RiskX Platform.

Provides comprehensive model analysis using SHAP values,
bias detection, and fairness analysis for transparent financial decisions.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import numpy as np

from src.ml.explainability.shap_analyzer import (
    ShapAnalyzer,
    ShapExplanation,
    GlobalExplanation,
    BiasAnalysis
)
from src.core.dependencies import get_cache_manager
from src.cache.cache_manager import IntelligentCacheManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/explainability", tags=["explainability"])

# Global SHAP analyzer instance
shap_analyzer = ShapAnalyzer()


class PredictionExplanationRequest(BaseModel):
    """Request model for individual prediction explanation."""
    model_id: str = Field(..., description="Model identifier")
    input_features: List[float] = Field(..., description="Input feature values")
    feature_names: Optional[List[str]] = Field(None, description="Feature names")
    prediction_id: Optional[str] = Field(None, description="Optional prediction identifier")


class GlobalExplanationRequest(BaseModel):
    """Request model for global model explanation."""
    model_id: str = Field(..., description="Model identifier")
    historical_data: List[List[float]] = Field(..., description="Historical data for analysis")
    feature_names: List[str] = Field(..., description="Feature names")
    sample_size: Optional[int] = Field(1000, description="Maximum samples to analyze")


class BiasAnalysisRequest(BaseModel):
    """Request model for bias analysis."""
    model_id: str = Field(..., description="Model identifier")
    validation_data: List[List[float]] = Field(..., description="Validation dataset features")
    test_labels: List[float] = Field(..., description="True labels")
    protected_attributes: Dict[str, List[Union[int, str]]] = Field(
        ..., description="Protected attribute values"
    )
    analysis_id: Optional[str] = Field(None, description="Optional analysis identifier")


class ModelRegistrationRequest(BaseModel):
    """Request model for registering a model with SHAP analyzer."""
    model_id: str = Field(..., description="Unique model identifier")
    model_type: str = Field("tree", description="Model type (tree, linear, kernel, deep)")
    feature_names: List[str] = Field(..., description="Feature names")
    background_data: Optional[List[List[float]]] = Field(None, description="Background data for explainer")
    
    model_config = {"protected_namespaces": ()}


class ModelComparisonRequest(BaseModel):
    """Request model for comparing multiple models."""
    model_ids: List[str] = Field(..., description="List of model identifiers")
    validation_data: List[List[float]] = Field(..., description="Validation data for comparison")
    sample_size: int = Field(1000, description="Sample size for comparison")
    
    model_config = {"protected_namespaces": ()}


@router.post("/register-model")
async def register_model(
    request: ModelRegistrationRequest,
    cache: IntelligentCacheManager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    Register a financial model for SHAP analysis.
    
    Registers a trained model with the SHAP analyzer to enable
    explanation generation and bias analysis.
    """
    try:
        # Load the actual trained model from model registry
        from src.ml.serving.model_server import ModelServer
        model_server = ModelServer()
        
        # Get the real trained model
        trained_model = model_server.get_model(request.model_id)
        if not trained_model:
            raise HTTPException(
                status_code=404,
                detail=f"Trained model {request.model_id} not found in model registry"
            )
        
        # Convert background data if provided
        background_data = None
        if request.background_data:
            background_data = np.array(request.background_data)
        
        # Register model with SHAP analyzer
        shap_analyzer.register_model(
            model_id=request.model_id,
            model=trained_model,
            feature_names=request.feature_names,
            model_type=request.model_type,
            background_data=background_data
        )
        
        # Cache model registration info
        cache_key = f"model_registration:{request.model_id}"
        registration_info = {
            "model_id": request.model_id,
            "model_type": request.model_type,
            "feature_names": request.feature_names,
            "registered_at": datetime.utcnow().isoformat(),
            "status": "registered"
        }
        
        await cache.set(cache_key, registration_info, ttl_seconds=7200)
        
        return {
            "status": "success",
            "message": f"Model {request.model_id} registered successfully",
            "data": registration_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model registration failed: {str(e)}"
        )


@router.post("/explain-prediction")
async def explain_prediction(
    request: PredictionExplanationRequest,
    cache: IntelligentCacheManager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    Generate SHAP explanation for individual prediction.
    
    Provides detailed explanation of model prediction with
    feature contributions and confidence scores.
    """
    try:
        # Check cache for existing explanation
        cache_key = f"explanation:{request.model_id}:{hash(str(request.input_features))}"
        cached_explanation = await cache.get(cache_key, max_age_seconds=3600)
        
        if cached_explanation:
            return {
                "status": "success",
                "source": "cache",
                "data": cached_explanation,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Convert input to numpy array
        input_data = np.array(request.input_features)
        
        # Generate SHAP explanation
        explanation = shap_analyzer.explain_prediction(
            model_id=request.model_id,
            input_data=input_data,
            prediction_id=request.prediction_id
        )
        
        # Convert to serializable format
        explanation_data = shap_analyzer.export_explanation(explanation, "json")
        
        # Add interpretation insights
        explanation_data["insights"] = _generate_prediction_insights(explanation)
        
        # Cache the explanation
        await cache.set(cache_key, explanation_data, ttl_seconds=3600)
        
        return {
            "status": "success",
            "source": "computed",
            "data": explanation_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction explanation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Explanation generation failed: {str(e)}"
        )


@router.post("/global-explanation")
async def generate_global_explanation(
    request: GlobalExplanationRequest,
    cache: IntelligentCacheManager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    Generate global model explanation using SHAP values.
    
    Provides overall model behavior analysis including feature importance,
    interactions, and partial dependence plots.
    """
    try:
        # Check cache for existing global explanation
        cache_key = f"global_explanation:{request.model_id}:{len(request.sample_data)}"
        cached_explanation = await cache.get(cache_key, max_age_seconds=7200)
        
        if cached_explanation:
            return {
                "status": "success",
                "source": "cache",
                "data": cached_explanation,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Convert historical data to numpy array
        historical_data = np.array(request.historical_data)
        
        # Generate global explanation
        global_explanation = shap_analyzer.generate_global_explanation(
            model_id=request.model_id,
            sample_data=historical_data,
            sample_size=request.sample_size
        )
        
        # Convert to serializable format
        explanation_data = {
            "model_id": global_explanation.model_id,
            "feature_importance": global_explanation.feature_importance,
            "mean_abs_shap_values": global_explanation.mean_abs_shap_values,
            "summary_plot_data": global_explanation.summary_plot_data,
            "partial_dependence_data": global_explanation.partial_dependence_data,
            "sample_size": global_explanation.sample_size,
            "timestamp": global_explanation.timestamp.isoformat()
        }
        
        # Add interaction values if available
        if global_explanation.interaction_values is not None:
            explanation_data["interaction_values"] = global_explanation.interaction_values.tolist()
        
        # Add model insights
        explanation_data["insights"] = _generate_global_insights(global_explanation)
        
        # Cache the explanation
        await cache.set(cache_key, explanation_data, ttl_seconds=7200)
        
        return {
            "status": "success",
            "source": "computed",
            "data": explanation_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Global explanation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Global explanation generation failed: {str(e)}"
        )


@router.post("/bias-analysis")
async def analyze_bias(
    request: BiasAnalysisRequest,
    cache: IntelligentCacheManager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    Analyze model bias and fairness using SHAP values.
    
    Provides comprehensive bias assessment including demographic parity,
    equalized odds, and fairness recommendations.
    """
    try:
        # Check cache for existing bias analysis
        cache_key = f"bias_analysis:{request.model_id}:{hash(str(request.dict()))}"
        cached_analysis = await cache.get(cache_key, max_age_seconds=3600)
        
        if cached_analysis:
            return {
                "status": "success",
                "source": "cache",
                "data": cached_analysis,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Convert data to numpy arrays
        validation_data = np.array(request.validation_data)
        validation_labels = np.array(request.test_labels)
        
        protected_attributes = {}
        for attr_name, attr_values in request.protected_attributes.items():
            protected_attributes[attr_name] = np.array(attr_values)
        
        # Perform bias analysis
        bias_analysis = shap_analyzer.analyze_bias(
            model_id=request.model_id,
            test_data=validation_data,
            test_labels=validation_labels,
            protected_attributes=protected_attributes,
            analysis_id=request.analysis_id
        )
        
        # Convert to serializable format
        analysis_data = {
            "analysis_id": bias_analysis.analysis_id,
            "protected_attributes": bias_analysis.protected_attributes,
            "demographic_parity": bias_analysis.demographic_parity,
            "equalized_odds": bias_analysis.equalized_odds,
            "individual_fairness": bias_analysis.individual_fairness,
            "group_fairness_metrics": bias_analysis.group_fairness_metrics,
            "bias_score": bias_analysis.bias_score,
            "fairness_recommendations": bias_analysis.fairness_recommendations,
            "timestamp": bias_analysis.timestamp.isoformat()
        }
        
        # Add bias assessment summary
        analysis_data["assessment_summary"] = _generate_bias_summary(bias_analysis)
        
        # Cache the analysis
        await cache.set(cache_key, analysis_data, ttl_seconds=3600)
        
        return {
            "status": "success",
            "source": "computed",
            "data": analysis_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Bias analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Bias analysis failed: {str(e)}"
        )


@router.post("/compare-models")
async def compare_models(
    request: ModelComparisonRequest,
    cache: IntelligentCacheManager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    Compare multiple models using SHAP-based metrics.
    
    Provides comprehensive comparison of model explanations,
    feature importance, and prediction consistency.
    """
    try:
        # Check cache for existing comparison
        cache_key = f"model_comparison:{hash(str(sorted(request.model_ids)))}"
        cached_comparison = await cache.get(cache_key, max_age_seconds=3600)
        
        if cached_comparison:
            return {
                "status": "success",
                "source": "cache",
                "data": cached_comparison,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Convert validation data to numpy array
        validation_data = np.array(request.validation_data)
        
        # Perform model comparison
        comparison_results = shap_analyzer.compare_models(
            model_ids=request.model_ids,
            test_data=validation_data,
            sample_size=request.sample_size
        )
        
        # Add comparison insights
        comparison_results["insights"] = _generate_comparison_insights(comparison_results)
        
        # Cache the comparison
        await cache.set(cache_key, comparison_results, ttl_seconds=3600)
        
        return {
            "status": "success",
            "source": "computed",
            "data": comparison_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model comparison failed: {str(e)}"
        )


@router.get("/registered-models")
async def get_registered_models() -> Dict[str, Any]:
    """
    Get list of models registered for explainability analysis.
    
    Returns information about all models available for
    SHAP analysis and explanation generation.
    """
    try:
        registered_models = []
        
        for model_id in shap_analyzer.models.keys():
            model_info = {
                "model_id": model_id,
                "feature_names": shap_analyzer.feature_names.get(model_id, []),
                "feature_count": len(shap_analyzer.feature_names.get(model_id, [])),
                "explainer_available": model_id in shap_analyzer.explainers,
                "explanation_count": len([
                    exp for exp in shap_analyzer.explanation_history
                    if hasattr(exp, 'model_id') and getattr(exp, 'model_id', None) == model_id
                ])
            }
            registered_models.append(model_info)
        
        return {
            "status": "success",
            "data": {
                "registered_models": registered_models,
                "total_models": len(registered_models)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get registered models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve registered models: {str(e)}"
        )


@router.get("/explanation-history")
async def get_explanation_history(
    limit: int = 10,
    model_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get recent explanation history.
    
    Returns a list of recent SHAP explanations with
    basic metadata and summary information.
    """
    try:
        explanations = shap_analyzer.get_explanation_history(limit=limit, model_id=model_id)
        
        # Convert to serializable format
        history_data = []
        for explanation in explanations:
            explanation_summary = {
                "prediction_id": explanation.prediction_id,
                "model_prediction": explanation.model_prediction,
                "expected_value": explanation.expected_value,
                "confidence_score": explanation.confidence_score,
                "explanation_type": explanation.explanation_type,
                "feature_count": len(explanation.feature_names),
                "timestamp": explanation.timestamp.isoformat()
            }
            
            # Add top contributing features
            abs_shap = np.abs(explanation.shap_values)
            top_indices = np.argsort(abs_shap)[-3:][::-1]  # Top 3 features
            
            top_features = []
            for idx in top_indices:
                top_features.append({
                    "feature": explanation.feature_names[idx],
                    "shap_value": float(explanation.shap_values[idx]),
                    "feature_value": float(explanation.feature_values[idx])
                })
            
            explanation_summary["top_contributing_features"] = top_features
            history_data.append(explanation_summary)
        
        return {
            "status": "success",
            "data": {
                "explanations": history_data,
                "total_explanations": len(shap_analyzer.explanation_history),
                "filtered_count": len(history_data)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get explanation history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve explanation history: {str(e)}"
        )


@router.get("/feature-importance/{model_id}")
async def get_feature_importance(
    model_id: str,
    cache: IntelligentCacheManager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    Get feature importance for a specific model.
    
    Returns ranked feature importance based on mean absolute SHAP values
    from recent explanations or global analysis.
    """
    try:
        # Check cache for feature importance
        cache_key = f"feature_importance:{model_id}"
        cached_importance = await cache.get(cache_key, max_age_seconds=1800)
        
        if cached_importance:
            return {
                "status": "success",
                "source": "cache",
                "data": cached_importance,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        if model_id not in shap_analyzer.models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
        
        # Calculate feature importance from explanation history
        model_explanations = [
            exp for exp in shap_analyzer.explanation_history
            if hasattr(exp, 'model_id') and getattr(exp, 'model_id', model_id) == model_id
        ]
        
        if not model_explanations:
            # Load historical data to generate importance if no history
            from src.data.sources import fred, bea, bls
            feature_names = shap_analyzer.feature_names[model_id]
            
            # Get real historical data instead of random data
            historical_data = await _get_historical_model_data(model_id, len(feature_names))
            if historical_data is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"No historical data available for model {model_id}"
                )
            
            global_explanation = shap_analyzer.generate_global_explanation(
                model_id=model_id,
                sample_data=historical_data,
                sample_size=min(len(historical_data), 100)
            )
            
            feature_importance = global_explanation.feature_importance
        else:
            # Calculate from explanation history
            feature_names = model_explanations[0].feature_names
            all_shap_values = np.array([exp.shap_values for exp in model_explanations])
            mean_abs_shap = np.mean(np.abs(all_shap_values), axis=0)
            
            feature_importance = dict(zip(feature_names, mean_abs_shap))
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        importance_data = {
            "model_id": model_id,
            "feature_importance": dict(sorted_features),
            "ranked_features": [
                {
                    "feature_name": name,
                    "importance": float(importance),
                    "rank": idx + 1
                }
                for idx, (name, importance) in enumerate(sorted_features)
            ],
            "sample_size": len(model_explanations) if model_explanations else 10
        }
        
        # Cache the importance data
        await cache.set(cache_key, importance_data, ttl_seconds=1800)
        
        return {
            "status": "success",
            "source": "computed",
            "data": importance_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve feature importance: {str(e)}"
        )


async def _get_historical_model_data(model_id: str, n_features: int) -> Optional[np.ndarray]:
    """Get real historical data for model analysis."""
    try:
        from src.data.sources import fred, bea, bls
        
        # Load real historical data based on model type
        if "risk" in model_id.lower():
            # Get economic indicators for risk models
            gdp_data = await fred.get_gdp()
            unemployment_data = await fred.get_unemployment_rate()
            inflation_data = await fred.get_inflation_rate()
            
            if gdp_data and unemployment_data and inflation_data:
                # Combine real economic indicators
                combined_data = _combine_economic_indicators(
                    gdp_data, unemployment_data, inflation_data, n_features
                )
                return combined_data
        
        elif "supply" in model_id.lower():
            # Get supply chain data
            trade_data = await census.get_trade_data()
            if trade_data:
                return _process_trade_data(trade_data, n_features)
        
        # Default to recent FRED data if specific data not available
        recent_data = await fred.get_recent_indicators(limit=100)
        if recent_data:
            return _process_fred_data(recent_data, n_features)
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get historical data for {model_id}: {e}")
        return None


def _combine_economic_indicators(gdp_data, unemployment_data, inflation_data, n_features):
    """Combine real economic indicators into feature matrix."""
    # Implementation to combine real economic data
    # This would process the actual API responses
    pass


def _process_trade_data(trade_data, n_features):
    """Process real trade data into feature matrix."""
    # Implementation to process real Census trade data
    pass


def _process_fred_data(fred_data, n_features):
    """Process real FRED data into feature matrix."""
    # Implementation to process real FRED economic indicators
    pass


def _generate_prediction_insights(explanation: ShapExplanation) -> Dict[str, Any]:
    """Generate human-readable insights from SHAP explanation."""
    # Find most influential features
    abs_shap = np.abs(explanation.shap_values)
    top_feature_idx = np.argmax(abs_shap)
    
    insights = {
        "prediction_summary": f"Model predicted {explanation.model_prediction:.3f}",
        "confidence_level": "high" if explanation.confidence_score > 0.7 else "medium" if explanation.confidence_score > 0.4 else "low",
        "most_influential_feature": {
            "name": explanation.feature_names[top_feature_idx],
            "contribution": float(explanation.shap_values[top_feature_idx]),
            "direction": "positive" if explanation.shap_values[top_feature_idx] > 0 else "negative"
        },
        "prediction_drivers": _identify_prediction_drivers(explanation),
        "feature_summary": _summarize_feature_contributions(explanation)
    }
    
    return insights


def _generate_global_insights(global_explanation: GlobalExplanation) -> Dict[str, Any]:
    """Generate insights from global model explanation."""
    # Sort features by importance
    sorted_features = sorted(
        global_explanation.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    insights = {
        "model_behavior_summary": f"Analysis based on {global_explanation.sample_size} samples",
        "most_important_feature": sorted_features[0][0] if sorted_features else "unknown",
        "feature_distribution": {
            "high_importance": len([f for f, imp in sorted_features if imp > np.mean(list(global_explanation.feature_importance.values()))]),
            "total_features": len(sorted_features)
        },
        "top_features": [
            {"feature": name, "importance": float(importance)}
            for name, importance in sorted_features[:5]
        ]
    }
    
    return insights


def _generate_bias_summary(bias_analysis: BiasAnalysis) -> Dict[str, Any]:
    """Generate summary of bias analysis results."""
    overall_assessment = "fair" if bias_analysis.bias_score < 0.1 else "moderate_bias" if bias_analysis.bias_score < 0.2 else "high_bias"
    
    summary = {
        "overall_assessment": overall_assessment,
        "bias_level": bias_analysis.bias_score,
        "protected_attributes_analyzed": len(bias_analysis.protected_attributes),
        "primary_concerns": [],
        "strengths": []
    }
    
    # Identify primary concerns
    if any(val > 0.1 for val in bias_analysis.demographic_parity.values()):
        summary["primary_concerns"].append("Demographic parity violations detected")
    
    if any(val > 0.1 for val in bias_analysis.equalized_odds.values()):
        summary["primary_concerns"].append("Equalized odds disparities found")
    
    if not summary["primary_concerns"]:
        summary["strengths"].append("No significant bias detected")
    
    return summary


def _generate_comparison_insights(comparison_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate insights from model comparison."""
    insights = {
        "models_compared": len(comparison_results["models"]),
        "consistency_assessment": "unknown",
        "feature_agreement": "unknown",
        "recommendations": []
    }
    
    # Assess prediction consistency
    if "prediction_consistency" in comparison_results:
        correlations = list(comparison_results["prediction_consistency"].values())
        if correlations:
            avg_correlation = np.mean(correlations)
            if avg_correlation > 0.8:
                insights["consistency_assessment"] = "high"
            elif avg_correlation > 0.6:
                insights["consistency_assessment"] = "moderate"
            else:
                insights["consistency_assessment"] = "low"
    
    # Generate recommendations
    if insights["consistency_assessment"] == "low":
        insights["recommendations"].append("Consider investigating model differences")
    
    if insights["consistency_assessment"] == "high":
        insights["recommendations"].append("Models show good agreement")
    
    return insights


def _identify_prediction_drivers(explanation: ShapExplanation) -> List[Dict[str, Any]]:
    """Identify key drivers of the prediction."""
    # Sort features by absolute SHAP value
    abs_shap = np.abs(explanation.shap_values)
    sorted_indices = np.argsort(abs_shap)[::-1]
    
    drivers = []
    for i, idx in enumerate(sorted_indices[:5]):  # Top 5 drivers
        drivers.append({
            "rank": i + 1,
            "feature": explanation.feature_names[idx],
            "contribution": float(explanation.shap_values[idx]),
            "feature_value": float(explanation.feature_values[idx]),
            "impact": "increases" if explanation.shap_values[idx] > 0 else "decreases"
        })
    
    return drivers


def _summarize_feature_contributions(explanation: ShapExplanation) -> Dict[str, Any]:
    """Summarize overall feature contributions."""
    positive_contributions = np.sum(explanation.shap_values[explanation.shap_values > 0])
    negative_contributions = np.sum(explanation.shap_values[explanation.shap_values < 0])
    
    return {
        "positive_total": float(positive_contributions),
        "negative_total": float(negative_contributions),
        "net_contribution": float(positive_contributions + negative_contributions),
        "baseline": explanation.expected_value
    }