"""Add analytics tables for page views, events, and feedback

Revision ID: 002_analytics
Revises: 001_initial
Create Date: 2024-12-02 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import Text, JSON, DateTime

# revision identifiers, used by Alembic.
revision = '002_analytics'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create page_views table
    op.create_table('page_views',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('path', sa.String(500), nullable=True),
        sa.Column('timestamp', DateTime(), nullable=True),
        sa.Column('user_agent', Text(), nullable=True),
        sa.Column('referrer', sa.String(500), nullable=True),
        sa.Column('viewport', sa.String(100), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_page_views_id'), 'page_views', ['id'], unique=False)
    op.create_index(op.f('ix_page_views_path'), 'page_views', ['path'], unique=False)
    op.create_index(op.f('ix_page_views_timestamp'), 'page_views', ['timestamp'], unique=False)

    # Create user_events table
    op.create_table('user_events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('event_name', sa.String(255), nullable=True),
        sa.Column('event_data', JSON(), nullable=True),
        sa.Column('timestamp', DateTime(), nullable=True),
        sa.Column('path', sa.String(500), nullable=True),
        sa.Column('user_session', sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_events_id'), 'user_events', ['id'], unique=False)
    op.create_index(op.f('ix_user_events_event_name'), 'user_events', ['event_name'], unique=False)
    op.create_index(op.f('ix_user_events_timestamp'), 'user_events', ['timestamp'], unique=False)

    # Create user_feedback table
    op.create_table('user_feedback',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('page', sa.String(500), nullable=True),
        sa.Column('rating', sa.Integer(), nullable=True),
        sa.Column('comment', Text(), nullable=True),
        sa.Column('category', sa.String(100), nullable=True),
        sa.Column('timestamp', DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_feedback_id'), 'user_feedback', ['id'], unique=False)
    op.create_index(op.f('ix_user_feedback_page'), 'user_feedback', ['page'], unique=False)
    op.create_index(op.f('ix_user_feedback_timestamp'), 'user_feedback', ['timestamp'], unique=False)


def downgrade() -> None:
    op.drop_table('user_feedback')
    op.drop_table('user_events')
    op.drop_table('page_views')