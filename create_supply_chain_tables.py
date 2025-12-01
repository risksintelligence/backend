#!/usr/bin/env python3
"""
Database Migration Script - Supply Chain Risk Models
Creates all new supply chain database tables for comprehensive risk analysis.
"""

import logging
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.db import engine, Base
from app.models import (
    # Existing models
    ObservationModel, SubmissionModel, JudgingLogModel, TransparencyLogModel,
    ModelMetadataModel, UserMetrics, PageView, UserEvent, UserFeedback,
    CommunityUser, CommunityInsight, InsightComment, InsightLike,
    WeeklyBrief, WeeklyBriefSubscription,
    
    # New supply chain models
    SupplyChainNode, SupplyChainRelationship, CascadeEvent,
    SectorVulnerabilityAssessment, SectorVulnerability,
    TimelineCascadeEvent, SPGlobalIntelligence, ResilienceMetric,
    RealTimeDataFeed, DataRefreshLog, ACLEDEvent,
    ComtradeTradeFlow, WTOTradeStatistic
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_tables():
    """Create all database tables."""
    try:
        logger.info("Starting database table creation...")
        
        # Create all tables defined in models
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ Successfully created all database tables!")
        
        # Log the tables that were created/verified
        table_names = [
            # Supply Chain Risk Tables
            'supply_chain_nodes', 'supply_chain_relationships', 'cascade_events',
            'sector_vulnerability_assessments', 'sector_vulnerabilities',
            'timeline_cascade_events', 'sp_global_intelligence', 'resilience_metrics',
            
            # Data Integration Tables
            'realtime_data_feeds', 'data_refresh_logs', 'acled_events',
            'comtrade_trade_flows', 'wto_trade_statistics',
            
            # Existing Tables (verified)
            'observations', 'submissions', 'judging_logs', 'transparency_logs',
            'model_metadata', 'user_metrics', 'page_views', 'user_events',
            'user_feedback', 'community_users', 'community_insights',
            'insight_comments', 'insight_likes', 'weekly_briefs', 'weekly_brief_subscriptions'
        ]
        
        logger.info(f"📊 Database schema includes {len(table_names)} tables:")
        for table in sorted(table_names):
            logger.info(f"  ✓ {table}")
        
        logger.info("\n🎯 New Supply Chain Features Enabled:")
        logger.info("  • Supply chain network modeling and relationship tracking")
        logger.info("  • Cascade event detection and impact analysis")
        logger.info("  • Sector-specific vulnerability assessments")
        logger.info("  • Timeline visualization for supply chain disruptions")
        logger.info("  • S&P Global intelligence integration")
        logger.info("  • Real-time data feeds and refresh monitoring")
        logger.info("  • Resilience metrics and scoring")
        logger.info("  • External data integration (ACLED, Comtrade, WTO)")
        
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {e}")
        raise

def verify_tables():
    """Verify that tables were created successfully."""
    try:
        from sqlalchemy import inspect
        
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        required_tables = [
            'supply_chain_nodes', 'supply_chain_relationships', 'cascade_events',
            'sector_vulnerability_assessments', 'sector_vulnerabilities',
            'timeline_cascade_events', 'sp_global_intelligence', 'resilience_metrics',
            'realtime_data_feeds', 'data_refresh_logs', 'acled_events',
            'comtrade_trade_flows', 'wto_trade_statistics'
        ]
        
        missing_tables = [table for table in required_tables if table not in existing_tables]
        
        if missing_tables:
            logger.warning(f"⚠️  Missing tables: {missing_tables}")
            return False
        else:
            logger.info("✅ All required supply chain tables are present!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to verify tables: {e}")
        return False

def show_table_info():
    """Show detailed information about the new tables."""
    try:
        from sqlalchemy import inspect
        
        inspector = inspect(engine)
        
        supply_chain_tables = [
            'supply_chain_nodes', 'supply_chain_relationships', 'cascade_events',
            'sector_vulnerability_assessments', 'sector_vulnerabilities',
            'timeline_cascade_events', 'sp_global_intelligence', 'resilience_metrics',
            'realtime_data_feeds', 'data_refresh_logs'
        ]
        
        logger.info("\n📋 Supply Chain Table Details:")
        for table in supply_chain_tables:
            if table in inspector.get_table_names():
                columns = inspector.get_columns(table)
                indexes = inspector.get_indexes(table)
                logger.info(f"\n  🗄️  {table.upper()}:")
                logger.info(f"     Columns: {len(columns)}")
                logger.info(f"     Indexes: {len(indexes)}")
                
                # Show key columns
                key_columns = [col['name'] for col in columns if any(
                    keyword in col['name'].lower() 
                    for keyword in ['id', 'score', 'risk', 'date', 'status']
                )][:5]  # Show first 5 key columns
                if key_columns:
                    logger.info(f"     Key columns: {', '.join(key_columns)}")
                
    except Exception as e:
        logger.error(f"Failed to show table info: {e}")

if __name__ == "__main__":
    logger.info("🚀 Supply Chain Risk Database Migration")
    logger.info("=" * 50)
    
    try:
        # Create tables
        create_tables()
        
        # Verify creation
        if verify_tables():
            # Show table information
            show_table_info()
            
            logger.info("\n🎉 Database migration completed successfully!")
            logger.info("Supply chain risk analysis features are now available.")
        else:
            logger.error("❌ Table verification failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)