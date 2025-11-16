"""Initial schema with all Phase A models

Revision ID: 001_initial
Revises: 
Create Date: 2024-11-16 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create observations table with data lineage fields
    op.create_table('observations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('series_id', sa.String(), nullable=True),
        sa.Column('observed_at', sa.DateTime(), nullable=True),
        sa.Column('value', sa.Float(), nullable=True),
        # Data lineage fields per architecture requirements
        sa.Column('source', sa.String(), nullable=True),
        sa.Column('source_url', sa.String(), nullable=True),
        sa.Column('fetched_at', sa.DateTime(), nullable=True),
        sa.Column('checksum', sa.String(), nullable=True),
        sa.Column('derivation_flag', sa.String(), nullable=True),
        # TTL tracking for cache management
        sa.Column('soft_ttl', sa.Integer(), nullable=True),
        sa.Column('hard_ttl', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_observations_id'), 'observations', ['id'], unique=False)
    op.create_index(op.f('ix_observations_series_id'), 'observations', ['series_id'], unique=False)
    op.create_index(op.f('ix_observations_source'), 'observations', ['source'], unique=False)
    op.create_index(op.f('ix_observations_fetched_at'), 'observations', ['fetched_at'], unique=False)

    # Create submissions table
    op.create_table('submissions',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('author', sa.String(), nullable=False),
        sa.Column('author_email', sa.String(), nullable=False),
        sa.Column('content_url', sa.String(), nullable=True),
        sa.Column('submission_type', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('meta_data', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_submissions_id'), 'submissions', ['id'], unique=False)

    # Create judging_logs table
    op.create_table('judging_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('submission_id', sa.String(), nullable=False),
        sa.Column('judge_id', sa.String(), nullable=False),
        sa.Column('action', sa.String(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('meta_data', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_judging_logs_id'), 'judging_logs', ['id'], unique=False)

    # Create transparency_logs table
    op.create_table('transparency_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('event_type', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('meta_data', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_transparency_logs_id'), 'transparency_logs', ['id'], unique=False)

    # Create model_metadata table
    op.create_table('model_metadata',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('version', sa.String(), nullable=False),
        sa.Column('trained_at', sa.DateTime(), nullable=False),
        sa.Column('training_window_start', sa.DateTime(), nullable=False),
        sa.Column('training_window_end', sa.DateTime(), nullable=False),
        sa.Column('performance_metrics', sa.JSON(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('file_path', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_name')
    )
    op.create_index(op.f('ix_model_metadata_id'), 'model_metadata', ['id'], unique=False)


def downgrade() -> None:
    op.drop_table('model_metadata')
    op.drop_table('transparency_logs')
    op.drop_table('judging_logs')
    op.drop_table('submissions')
    op.drop_table('observations')