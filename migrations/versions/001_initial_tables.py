"""Initial database tables

Revision ID: 001
Revises: 
Create Date: 2025-10-23 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create cache_entries table
    op.create_table('cache_entries',
        sa.Column('cache_key', sa.String(length=255), nullable=False),
        sa.Column('data', sa.Text(), nullable=False),
        sa.Column('ttl_seconds', sa.Integer(), nullable=True),
        sa.Column('data_source', sa.String(length=50), nullable=True),
        sa.Column('cache_tier', sa.String(length=10), nullable=True),
        sa.Column('size_bytes', sa.Integer(), nullable=True),
        sa.Column('cached_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('accessed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('access_count', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('cache_key')
    )
    op.create_index('idx_cache_entries_expires', 'cache_entries', ['expires_at'], unique=False)
    op.create_index('idx_cache_entries_source', 'cache_entries', ['data_source'], unique=False)
    op.create_index(op.f('ix_cache_entries_cache_key'), 'cache_entries', ['cache_key'], unique=False)
    op.create_index(op.f('ix_cache_entries_cached_at'), 'cache_entries', ['cached_at'], unique=False)
    op.create_index(op.f('ix_cache_entries_expires_at'), 'cache_entries', ['expires_at'], unique=False)

    # Create users table
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=200), nullable=True),
        sa.Column('organization', sa.String(length=200), nullable=True),
        sa.Column('role', sa.String(length=50), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('is_superuser', sa.Boolean(), nullable=False),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=True),
        sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('preferences', sa.JSON(), nullable=True),
        sa.Column('api_key', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_users_active', 'users', ['is_active'], unique=False)
    op.create_index('idx_users_role', 'users', ['role'], unique=False)
    op.create_index(op.f('ix_users_api_key'), 'users', ['api_key'], unique=True)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)

    # Create risk_scores table
    op.create_table('risk_scores',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('overall_score', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('trend', sa.String(length=20), nullable=False),
        sa.Column('economic_score', sa.Float(), nullable=False),
        sa.Column('market_score', sa.Float(), nullable=False),
        sa.Column('geopolitical_score', sa.Float(), nullable=False),
        sa.Column('technical_score', sa.Float(), nullable=False),
        sa.Column('data_sources', sa.JSON(), nullable=True),
        sa.Column('calculation_method', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_risk_scores_overall', 'risk_scores', ['overall_score'], unique=False)
    op.create_index('idx_risk_scores_timestamp', 'risk_scores', ['timestamp'], unique=False)
    op.create_index('idx_risk_scores_trend', 'risk_scores', ['trend'], unique=False)
    op.create_index(op.f('ix_risk_scores_id'), 'risk_scores', ['id'], unique=False)

    # Create risk_factors table
    op.create_table('risk_factors',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('current_value', sa.Float(), nullable=False),
        sa.Column('current_score', sa.Float(), nullable=False),
        sa.Column('impact_level', sa.String(length=20), nullable=False),
        sa.Column('weight', sa.Float(), nullable=False),
        sa.Column('threshold_low', sa.Float(), nullable=True),
        sa.Column('threshold_high', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('data_source', sa.String(length=50), nullable=False),
        sa.Column('series_id', sa.String(length=100), nullable=True),
        sa.Column('update_frequency', sa.String(length=20), nullable=True),
        sa.Column('last_updated', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_risk_factors_active', 'risk_factors', ['is_active'], unique=False)
    op.create_index('idx_risk_factors_category', 'risk_factors', ['category'], unique=False)
    op.create_index('idx_risk_factors_source', 'risk_factors', ['data_source'], unique=False)
    op.create_index(op.f('ix_risk_factors_id'), 'risk_factors', ['id'], unique=False)
    op.create_index(op.f('ix_risk_factors_name'), 'risk_factors', ['name'], unique=True)

    # Create economic_indicators table
    op.create_table('economic_indicators',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('series_id', sa.String(length=100), nullable=False),
        sa.Column('source', sa.String(length=50), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('units', sa.String(length=100), nullable=False),
        sa.Column('frequency', sa.String(length=20), nullable=False),
        sa.Column('observation_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('period', sa.String(length=20), nullable=True),
        sa.Column('seasonal_adjustment', sa.String(length=50), nullable=True),
        sa.Column('revision_status', sa.String(length=20), nullable=True),
        sa.Column('period_change', sa.Float(), nullable=True),
        sa.Column('year_over_year_change', sa.Float(), nullable=True),
        sa.Column('fetched_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_economic_indicators_frequency', 'economic_indicators', ['frequency'], unique=False)
    op.create_index('idx_economic_indicators_series_date', 'economic_indicators', ['series_id', 'observation_date'], unique=False)
    op.create_index('idx_economic_indicators_source_date', 'economic_indicators', ['source', 'observation_date'], unique=False)
    op.create_index(op.f('ix_economic_indicators_id'), 'economic_indicators', ['id'], unique=False)
    op.create_index(op.f('ix_economic_indicators_observation_date'), 'economic_indicators', ['observation_date'], unique=False)
    op.create_index(op.f('ix_economic_indicators_series_id'), 'economic_indicators', ['series_id'], unique=False)
    op.create_index(op.f('ix_economic_indicators_source'), 'economic_indicators', ['source'], unique=False)

    # Create alerts table
    op.create_table('alerts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('alert_type', sa.String(length=50), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('triggered_by', sa.String(length=100), nullable=True),
        sa.Column('threshold_value', sa.Float(), nullable=True),
        sa.Column('current_value', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('acknowledged_by', sa.Integer(), nullable=True),
        sa.Column('acknowledged_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('alert_metadata', sa.JSON(), nullable=True),
        sa.Column('triggered_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_alerts_status', 'alerts', ['status'], unique=False)
    op.create_index('idx_alerts_triggered', 'alerts', ['triggered_at'], unique=False)
    op.create_index('idx_alerts_type_severity', 'alerts', ['alert_type', 'severity'], unique=False)
    op.create_index(op.f('ix_alerts_alert_type'), 'alerts', ['alert_type'], unique=False)
    op.create_index(op.f('ix_alerts_id'), 'alerts', ['id'], unique=False)
    op.create_index(op.f('ix_alerts_severity'), 'alerts', ['severity'], unique=False)
    op.create_index(op.f('ix_alerts_triggered_at'), 'alerts', ['triggered_at'], unique=False)

    # Create system_metrics table
    op.create_table('system_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_type', sa.String(length=50), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('unit', sa.String(length=50), nullable=True),
        sa.Column('component', sa.String(length=100), nullable=False),
        sa.Column('environment', sa.String(length=20), nullable=True),
        sa.Column('instance_id', sa.String(length=100), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('metric_metadata', sa.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_system_metrics_name_component', 'system_metrics', ['metric_name', 'component'], unique=False)
    op.create_index('idx_system_metrics_timestamp', 'system_metrics', ['timestamp'], unique=False)
    op.create_index('idx_system_metrics_type', 'system_metrics', ['metric_type'], unique=False)
    op.create_index(op.f('ix_system_metrics_component'), 'system_metrics', ['component'], unique=False)
    op.create_index(op.f('ix_system_metrics_id'), 'system_metrics', ['id'], unique=False)
    op.create_index(op.f('ix_system_metrics_metric_name'), 'system_metrics', ['metric_name'], unique=False)
    op.create_index(op.f('ix_system_metrics_timestamp'), 'system_metrics', ['timestamp'], unique=False)


def downgrade() -> None:
    op.drop_table('system_metrics')
    op.drop_table('alerts')
    op.drop_table('economic_indicators')
    op.drop_table('risk_factors')
    op.drop_table('risk_scores')
    op.drop_table('users')
    op.drop_table('cache_entries')