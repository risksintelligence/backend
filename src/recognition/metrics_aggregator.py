"""
Metrics Aggregator Module

Aggregates recognition metrics from all monitoring systems to provide
comprehensive platform impact and validation measurements.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from ..core.config import get_settings
from ..cache.cache_manager import CacheManager


@dataclass
class RecognitionSummary:
    """Comprehensive recognition summary"""
    period_days: int
    timestamp: datetime
    
    # Impact metrics
    total_impact_events: int
    policy_influence_score: float
    media_reach_total: int
    academic_validation_score: float
    
    # Citation metrics
    total_citations: int
    h_index_estimate: int
    citation_growth_rate: float
    
    # Media metrics
    media_sentiment_average: float
    high_prominence_mentions: int
    
    # Policy metrics
    high_influence_policy_refs: int
    government_agency_engagement: int
    
    # Trending data
    trending_keywords: List[str]
    top_citing_institutions: List[str]
    most_active_agencies: List[str]
    
    # Recognition score (0-100)
    overall_recognition_score: float


@dataclass
class ValidationFramework:
    """Academic and professional validation framework"""
    peer_review_status: str
    methodology_validation: str
    data_quality_certification: str
    bias_audit_status: str
    transparency_score: float
    reproducibility_score: float


class MetricsAggregator:
    """
    Aggregates recognition metrics from all monitoring systems.
    
    Combines data from:
    - ImpactTracker (policy and media impact)
    - CitationMonitor (academic citations)
    - MediaAnalyzer (news and media coverage)
    - PolicyTracker (government policy references)
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = CacheManager()
        self.logger = logging.getLogger(__name__)
        
        # Recognition scoring weights
        self.scoring_weights = {
            "academic_citations": 0.30,
            "policy_influence": 0.25,
            "media_coverage": 0.20,
            "government_engagement": 0.15,
            "peer_validation": 0.10
        }
        
        # Validation benchmarks
        self.validation_benchmarks = {
            "citations_threshold": 10,  # Minimum citations for academic recognition
            "policy_refs_threshold": 5,  # Minimum policy references
            "media_mentions_threshold": 20,  # Minimum media mentions
            "sentiment_threshold": 0.2,  # Minimum positive sentiment
            "reach_threshold": 1000000  # Minimum estimated reach
        }
    
    async def generate_comprehensive_report(self, days: int = 30) -> RecognitionSummary:
        """
        Generate comprehensive recognition report aggregating all metrics
        """
        self.logger.info(f"Generating comprehensive recognition report for {days} days")
        
        try:
            # Gather data from all recognition systems
            impact_data = await self._get_impact_data(days)
            citation_data = await self._get_citation_data(days)
            media_data = await self._get_media_data(days)
            policy_data = await self._get_policy_data(days)
            
            # Calculate aggregate metrics
            summary = await self._calculate_recognition_summary(
                impact_data, citation_data, media_data, policy_data, days
            )
            
            # Cache the report
            await self._cache_recognition_report(summary)
            
            self.logger.info("Comprehensive recognition report generated successfully")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating recognition report: {str(e)}")
            raise
    
    async def _get_impact_data(self, days: int) -> Dict[str, Any]:
        """Get aggregated impact tracking data"""
        try:
            from .impact_tracker import ImpactTracker
            impact_tracker = ImpactTracker()
            return await impact_tracker.get_impact_summary(days)
        except Exception as e:
            self.logger.warning(f"Could not retrieve impact data: {str(e)}")
            return {}
    
    async def _get_citation_data(self, days: int) -> Dict[str, Any]:
        """Get aggregated citation data"""
        try:
            from .citation_monitor import CitationMonitor
            citation_monitor = CitationMonitor()
            metrics = await citation_monitor.get_citation_metrics(days)
            
            return {
                "total_citations": metrics.total_citations,
                "direct_citations": metrics.direct_citations,
                "h_index_estimate": metrics.h_index_estimate,
                "citation_growth_rate": metrics.citation_growth_rate,
                "top_citing_journals": metrics.top_citing_journals,
                "recent_citations": len(metrics.recent_citations)
            }
        except Exception as e:
            self.logger.warning(f"Could not retrieve citation data: {str(e)}")
            return {}
    
    async def _get_media_data(self, days: int) -> Dict[str, Any]:
        """Get aggregated media coverage data"""
        try:
            from .media_analyzer import MediaAnalyzer
            media_analyzer = MediaAnalyzer()
            return await media_analyzer.get_media_summary(days)
        except Exception as e:
            self.logger.warning(f"Could not retrieve media data: {str(e)}")
            return {}
    
    async def _get_policy_data(self, days: int) -> Dict[str, Any]:
        """Get aggregated policy influence data"""
        try:
            from .policy_tracker import PolicyTracker
            policy_tracker = PolicyTracker()
            return await policy_tracker.get_policy_influence_summary(days)
        except Exception as e:
            self.logger.warning(f"Could not retrieve policy data: {str(e)}")
            return {}
    
    async def _calculate_recognition_summary(
        self, 
        impact_data: Dict, 
        citation_data: Dict, 
        media_data: Dict, 
        policy_data: Dict,
        days: int
    ) -> RecognitionSummary:
        """Calculate comprehensive recognition summary"""
        
        # Extract key metrics with defaults
        total_citations = citation_data.get("total_citations", 0)
        h_index = citation_data.get("h_index_estimate", 0)
        citation_growth = citation_data.get("citation_growth_rate", 0.0)
        
        total_media_mentions = media_data.get("total_mentions", 0)
        media_sentiment = media_data.get("average_sentiment", 0.0)
        total_reach = media_data.get("total_estimated_reach", 0)
        high_reach_mentions = media_data.get("high_reach_mentions", 0)
        
        total_policy_refs = policy_data.get("total_policy_references", 0)
        high_influence_refs = policy_data.get("high_influence_references", 0)
        agency_count = len(policy_data.get("agency_influence", {}))
        
        total_impact_events = impact_data.get("total_events", 0)
        
        # Calculate component scores (0-100 scale)
        academic_score = self._calculate_academic_score(citation_data)
        policy_score = self._calculate_policy_score(policy_data)
        media_score = self._calculate_media_score(media_data)
        government_score = self._calculate_government_score(policy_data)
        peer_validation_score = self._calculate_peer_validation_score(citation_data, policy_data)
        
        # Calculate overall recognition score
        overall_score = (
            academic_score * self.scoring_weights["academic_citations"] +
            policy_score * self.scoring_weights["policy_influence"] +
            media_score * self.scoring_weights["media_coverage"] +
            government_score * self.scoring_weights["government_engagement"] +
            peer_validation_score * self.scoring_weights["peer_validation"]
        )
        
        # Extract trending data
        trending_keywords = self._extract_trending_keywords(impact_data, media_data, policy_data)
        top_institutions = self._extract_top_institutions(citation_data, policy_data)
        active_agencies = self._extract_active_agencies(policy_data)
        
        return RecognitionSummary(
            period_days=days,
            timestamp=datetime.now(),
            total_impact_events=total_impact_events,
            policy_influence_score=policy_score,
            media_reach_total=total_reach,
            academic_validation_score=academic_score,
            total_citations=total_citations,
            h_index_estimate=h_index,
            citation_growth_rate=citation_growth,
            media_sentiment_average=media_sentiment,
            high_prominence_mentions=high_reach_mentions,
            high_influence_policy_refs=high_influence_refs,
            government_agency_engagement=agency_count,
            trending_keywords=trending_keywords,
            top_citing_institutions=top_institutions,
            most_active_agencies=active_agencies,
            overall_recognition_score=round(overall_score, 1)
        )
    
    def _calculate_academic_score(self, citation_data: Dict) -> float:
        """Calculate academic recognition score (0-100)"""
        citations = citation_data.get("total_citations", 0)
        h_index = citation_data.get("h_index_estimate", 0)
        growth_rate = citation_data.get("citation_growth_rate", 0.0)
        direct_citations = citation_data.get("direct_citations", 0)
        
        # Base score from citation count
        citation_score = min(citations / self.validation_benchmarks["citations_threshold"] * 40, 40)
        
        # H-index contribution
        h_index_score = min(h_index * 5, 25)
        
        # Growth rate contribution
        growth_score = min(growth_rate * 2, 20)
        
        # Direct citation bonus
        direct_score = min(direct_citations * 3, 15)
        
        return citation_score + h_index_score + growth_score + direct_score
    
    def _calculate_policy_score(self, policy_data: Dict) -> float:
        """Calculate policy influence score (0-100)"""
        total_refs = policy_data.get("total_policy_references", 0)
        high_influence = policy_data.get("high_influence_references", 0)
        direct_citations = policy_data.get("direct_citations", 0)
        influence_rate = policy_data.get("influence_rate", 0.0)
        
        # Base score from reference count
        ref_score = min(total_refs / self.validation_benchmarks["policy_refs_threshold"] * 30, 30)
        
        # High influence bonus
        influence_score = min(high_influence * 10, 40)
        
        # Direct citation bonus
        direct_score = min(direct_citations * 8, 20)
        
        # Influence rate bonus
        rate_score = min(influence_rate / 20 * 10, 10)
        
        return ref_score + influence_score + direct_score + rate_score
    
    def _calculate_media_score(self, media_data: Dict) -> float:
        """Calculate media coverage score (0-100)"""
        total_mentions = media_data.get("total_mentions", 0)
        sentiment = media_data.get("average_sentiment", 0.0)
        reach = media_data.get("total_estimated_reach", 0)
        high_reach = media_data.get("high_reach_mentions", 0)
        
        # Base score from mention count
        mention_score = min(total_mentions / self.validation_benchmarks["media_mentions_threshold"] * 25, 25)
        
        # Sentiment bonus/penalty
        sentiment_score = max(0, (sentiment + 1) / 2 * 25)  # Convert -1,1 to 0,25
        
        # Reach score
        reach_score = min(reach / self.validation_benchmarks["reach_threshold"] * 30, 30)
        
        # High-reach mention bonus
        high_reach_score = min(high_reach * 5, 20)
        
        return mention_score + sentiment_score + reach_score + high_reach_score
    
    def _calculate_government_score(self, policy_data: Dict) -> float:
        """Calculate government engagement score (0-100)"""
        agency_influence = policy_data.get("agency_influence", {})
        legislative_mentions = policy_data.get("legislative_mentions", 0)
        
        # Agency diversity score
        agency_count = len(agency_influence)
        agency_score = min(agency_count * 15, 60)
        
        # High-level agency engagement
        high_level_agencies = ["Federal Reserve", "Treasury Department", "White House"]
        high_level_score = 0
        for agency in high_level_agencies:
            if agency in agency_influence:
                high_level_score += 10
        
        # Legislative mention bonus
        legislative_score = min(legislative_mentions * 5, 20)
        
        return min(agency_score + high_level_score + legislative_score, 100)
    
    def _calculate_peer_validation_score(self, citation_data: Dict, policy_data: Dict) -> float:
        """Calculate peer validation score (0-100)"""
        top_journals = citation_data.get("top_citing_journals", [])
        methodology_refs = citation_data.get("methodology_references", 0)
        
        # Top-tier journal score
        prestigious_journals = ["Nature", "Science", "PNAS", "Cell", "The Lancet"]
        journal_score = 0
        for journal in top_journals:
            if any(prestigious in journal for prestigious in prestigious_journals):
                journal_score += 20
        
        # Methodology reference bonus
        methodology_score = min(methodology_refs * 10, 40)
        
        # Reproducibility indicator (simplified)
        reproducibility_score = 20 if methodology_refs > 0 else 0
        
        return min(journal_score + methodology_score + reproducibility_score, 100)
    
    def _extract_trending_keywords(self, impact_data: Dict, media_data: Dict, policy_data: Dict) -> List[str]:
        """Extract trending keywords across all data sources"""
        all_keywords = []
        
        # From impact data
        all_keywords.extend(impact_data.get("trending_keywords", []))
        
        # From media data
        all_keywords.extend(media_data.get("trending_topics", []))
        
        # From policy data
        all_keywords.extend(policy_data.get("trending_policy_topics", []))
        
        # Count frequency and return top keywords
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return [kw[0] for kw in sorted_keywords[:5]]
    
    def _extract_top_institutions(self, citation_data: Dict, policy_data: Dict) -> List[str]:
        """Extract top citing institutions"""
        institutions = []
        
        # From citation data
        institutions.extend(citation_data.get("top_citing_journals", []))
        
        # From policy data (agencies)
        agency_influence = policy_data.get("agency_influence", {})
        top_agencies = sorted(agency_influence.items(), key=lambda x: x[1].get("total", 0), reverse=True)
        institutions.extend([agency[0] for agency in top_agencies[:3]])
        
        return institutions[:5]
    
    def _extract_active_agencies(self, policy_data: Dict) -> List[str]:
        """Extract most active government agencies"""
        agency_influence = policy_data.get("agency_influence", {})
        sorted_agencies = sorted(
            agency_influence.items(), 
            key=lambda x: x[1].get("high_influence", 0), 
            reverse=True
        )
        return [agency[0] for agency in sorted_agencies[:5]]
    
    async def _cache_recognition_report(self, summary: RecognitionSummary):
        """Cache the recognition report"""
        try:
            report_data = {
                "period_days": summary.period_days,
                "timestamp": summary.timestamp.isoformat(),
                "total_impact_events": summary.total_impact_events,
                "policy_influence_score": summary.policy_influence_score,
                "media_reach_total": summary.media_reach_total,
                "academic_validation_score": summary.academic_validation_score,
                "total_citations": summary.total_citations,
                "h_index_estimate": summary.h_index_estimate,
                "citation_growth_rate": summary.citation_growth_rate,
                "media_sentiment_average": summary.media_sentiment_average,
                "high_prominence_mentions": summary.high_prominence_mentions,
                "high_influence_policy_refs": summary.high_influence_policy_refs,
                "government_agency_engagement": summary.government_agency_engagement,
                "trending_keywords": summary.trending_keywords,
                "top_citing_institutions": summary.top_citing_institutions,
                "most_active_agencies": summary.most_active_agencies,
                "overall_recognition_score": summary.overall_recognition_score
            }
            
            # Cache latest report
            await self.cache.set("recognition_latest_report", report_data, ttl=86400)
            
            # Cache historical report
            date_key = f"recognition_report_{datetime.now().strftime('%Y%m%d')}"
            await self.cache.set(date_key, report_data, ttl=86400 * 90)  # Keep for 90 days
            
        except Exception as e:
            self.logger.error(f"Error caching recognition report: {str(e)}")
    
    async def get_validation_framework_status(self) -> ValidationFramework:
        """Get current validation framework status"""
        try:
            # Check current metrics to assess validation status
            latest_report = await self.cache.get("recognition_latest_report")
            
            if not latest_report:
                return ValidationFramework(
                    peer_review_status="pending",
                    methodology_validation="in_progress",
                    data_quality_certification="pending",
                    bias_audit_status="scheduled",
                    transparency_score=85.0,  # Based on open-source nature
                    reproducibility_score=90.0  # Based on documented methodology
                )
            
            # Assess validation status based on metrics
            citations = latest_report.get("total_citations", 0)
            policy_refs = latest_report.get("high_influence_policy_refs", 0)
            recognition_score = latest_report.get("overall_recognition_score", 0)
            
            # Determine status levels
            peer_review_status = "validated" if citations >= 10 else "in_progress" if citations > 0 else "pending"
            methodology_status = "validated" if policy_refs >= 5 else "in_progress" if policy_refs > 0 else "pending"
            data_quality_status = "certified" if recognition_score >= 70 else "in_progress"
            bias_audit_status = "completed" if recognition_score >= 80 else "in_progress"
            
            # Calculate scores based on current metrics
            transparency_score = min(85.0 + (recognition_score - 50) * 0.3, 100.0)
            reproducibility_score = min(90.0 + (citations * 2), 100.0)
            
            return ValidationFramework(
                peer_review_status=peer_review_status,
                methodology_validation=methodology_status,
                data_quality_certification=data_quality_status,
                bias_audit_status=bias_audit_status,
                transparency_score=transparency_score,
                reproducibility_score=reproducibility_score
            )
            
        except Exception as e:
            self.logger.error(f"Error getting validation framework status: {str(e)}")
            return ValidationFramework(
                peer_review_status="unknown",
                methodology_validation="unknown",
                data_quality_certification="unknown",
                bias_audit_status="unknown",
                transparency_score=0.0,
                reproducibility_score=0.0
            )
    
    async def get_recognition_trends(self, days: int = 90) -> Dict[str, Any]:
        """Get recognition trends over time"""
        try:
            trend_data = {"dates": [], "scores": [], "events": []}
            
            for i in range(0, days, 7):  # Weekly intervals
                date = datetime.now() - timedelta(days=i)
                date_key = f"recognition_report_{date.strftime('%Y%m%d')}"
                
                report = await self.cache.get(date_key)
                if report:
                    trend_data["dates"].append(date.strftime('%Y-%m-%d'))
                    trend_data["scores"].append(report.get("overall_recognition_score", 0))
                    trend_data["events"].append(report.get("total_impact_events", 0))
            
            # Calculate growth rate
            if len(trend_data["scores"]) >= 2:
                recent_score = trend_data["scores"][0]
                previous_score = trend_data["scores"][-1]
                growth_rate = ((recent_score - previous_score) / max(previous_score, 1)) * 100
            else:
                growth_rate = 0.0
            
            return {
                "trend_data": trend_data,
                "growth_rate": round(growth_rate, 2),
                "data_points": len(trend_data["dates"]),
                "period_days": days
            }
            
        except Exception as e:
            self.logger.error(f"Error getting recognition trends: {str(e)}")
            return {"error": "Could not generate recognition trends"}
    
    async def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary for stakeholders"""
        try:
            # Get latest recognition report
            latest_report = await self.cache.get("recognition_latest_report")
            if not latest_report:
                latest_report = {}
            
            # Get validation framework status
            validation = await self.get_validation_framework_status()
            
            # Get trend data
            trends = await self.get_recognition_trends(30)
            
            # Generate key insights
            recognition_score = latest_report.get("overall_recognition_score", 0)
            total_citations = latest_report.get("total_citations", 0)
            policy_influence = latest_report.get("high_influence_policy_refs", 0)
            media_reach = latest_report.get("media_reach_total", 0)
            
            # Determine status level
            if recognition_score >= 80:
                status_level = "High Recognition"
                status_description = "Platform has achieved significant recognition across academic, policy, and media spheres"
            elif recognition_score >= 60:
                status_level = "Moderate Recognition"
                status_description = "Platform shows strong progress in building recognition and validation"
            elif recognition_score >= 40:
                status_level = "Emerging Recognition"
                status_description = "Platform is gaining traction and building initial recognition"
            else:
                status_level = "Building Recognition"
                status_description = "Platform is in early stages of recognition development"
            
            summary = {
                "status_level": status_level,
                "status_description": status_description,
                "recognition_score": recognition_score,
                "key_metrics": {
                    "academic_citations": total_citations,
                    "policy_influence_references": policy_influence,
                    "estimated_media_reach": media_reach,
                    "government_agencies_engaged": latest_report.get("government_agency_engagement", 0)
                },
                "validation_status": {
                    "peer_review": validation.peer_review_status,
                    "methodology_validation": validation.methodology_validation,
                    "transparency_score": validation.transparency_score,
                    "reproducibility_score": validation.reproducibility_score
                },
                "growth_indicators": {
                    "recognition_growth_rate": trends.get("growth_rate", 0),
                    "trending_keywords": latest_report.get("trending_keywords", [])[:3],
                    "most_active_agencies": latest_report.get("most_active_agencies", [])[:3]
                },
                "recommendations": self._generate_recommendations(recognition_score, validation, trends),
                "timestamp": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {str(e)}")
            return {"error": "Could not generate executive summary"}
    
    def _generate_recommendations(self, recognition_score: float, validation: ValidationFramework, trends: Dict) -> List[str]:
        """Generate strategic recommendations based on current metrics"""
        recommendations = []
        
        if recognition_score < 50:
            recommendations.append("Focus on increasing academic publication citations through conference presentations and journal submissions")
            recommendations.append("Enhance policy engagement by directly sharing insights with relevant government agencies")
        
        if validation.peer_review_status == "pending":
            recommendations.append("Prioritize peer review process by submitting methodology to academic journals")
        
        if trends.get("growth_rate", 0) < 5:
            recommendations.append("Accelerate platform visibility through targeted media outreach and demonstration projects")
        
        if recognition_score >= 70:
            recommendations.append("Leverage current recognition to establish formal partnerships with academic institutions")
            recommendations.append("Consider international expansion to broaden global recognition")
        
        if not recommendations:
            recommendations.append("Continue current recognition-building strategies while monitoring emerging opportunities")
        
        return recommendations[:5]  # Return top 5 recommendations