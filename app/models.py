from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from app.db import Base
import uuid

class ObservationModel(Base):
    __tablename__ = 'observations'
    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(String, index=True)
    observed_at = Column(DateTime)
    value = Column(Float)
    
    # Data lineage fields per architecture requirements
    source = Column(String, index=True)  # Provider name (fred, eia, etc.)
    source_url = Column(String)  # Original API endpoint
    fetched_at = Column(DateTime, index=True)  # When data was retrieved
    checksum = Column(String)  # Data integrity verification
    derivation_flag = Column(String)  # raw, derived, or blended
    
    # TTL tracking for cache management
    soft_ttl = Column(Integer)  # Seconds until background refresh
    hard_ttl = Column(Integer)  # Seconds until marked stale

class SubmissionModel(Base):
    __tablename__ = 'submissions'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    title = Column(String, nullable=False)
    summary = Column(Text)
    author = Column(String, nullable=False)
    author_email = Column(String, nullable=False)
    content_url = Column(String)
    submission_type = Column(String, nullable=False)  # analysis, lab, policy
    status = Column(String, default='pending')  # pending, approved, rejected
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime)
    meta_data = Column(JSON)

class JudgingLogModel(Base):
    __tablename__ = 'judging_logs'
    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(String, nullable=False)
    judge_id = Column(String, nullable=False)
    action = Column(String, nullable=False)  # approve, reject, comment
    notes = Column(Text)
    timestamp = Column(DateTime, nullable=False)
    meta_data = Column(JSON)

class TransparencyLogModel(Base):
    __tablename__ = 'transparency_logs'
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, nullable=False)  # data_update, model_retrain, system_change
    description = Column(Text)
    timestamp = Column(DateTime, nullable=False)
    meta_data = Column(JSON)

class ModelMetadataModel(Base):
    __tablename__ = 'model_metadata'
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False, unique=True)
    version = Column(String, nullable=False)
    trained_at = Column(DateTime, nullable=False)
    training_window_start = Column(DateTime, nullable=False)
    training_window_end = Column(DateTime, nullable=False)
    performance_metrics = Column(JSON)
    is_active = Column(Boolean, default=False)
    file_path = Column(String)  # Path to the .pkl file
    created_at = Column(DateTime, nullable=False)
