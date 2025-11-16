from datetime import datetime
from typing import Dict, List
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db import SessionLocal
from app.models import SubmissionModel
from app.services.transparency import record_update
from app.services.impact import update_snapshot


def add_submission(payload: Dict[str, str]) -> Dict[str, str]:
    """Add new submission to PostgreSQL database."""
    db = SessionLocal()
    try:
        submission = SubmissionModel(
            title=payload.get('title', 'Untitled'),
            summary=payload.get('summary', ''),
            author=payload.get('author', 'Anonymous'),
            author_email=payload.get('author_email', 'noreply@example.com'),
            content_url=payload.get('link', ''),
            submission_type=payload.get('mission', 'analysis'),
            status='pending',
            created_at=datetime.utcnow()
        )
        
        db.add(submission)
        db.commit()
        db.refresh(submission)
        
        record_update(f"New submission: {submission.title}")
        update_snapshot({"analyses": 0.01})
        
        return {
            "id": submission.id,
            "title": submission.title,
            "author": submission.author,
            "submitted_at": submission.created_at.isoformat() + 'Z',
            "status": submission.status,
            "mission": submission.submission_type,
            "link": submission.content_url
        }
        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def list_submissions() -> List[Dict[str, str]]:
    """List all submissions from PostgreSQL database."""
    db = SessionLocal()
    try:
        submissions = db.query(SubmissionModel).order_by(
            SubmissionModel.created_at.desc()
        ).all()
        
        return [
            {
                "id": sub.id,
                "title": sub.title,
                "author": sub.author,
                "submitted_at": sub.created_at.isoformat() + 'Z',
                "status": sub.status,
                "mission": sub.submission_type,
                "link": sub.content_url or '',
                "summary": sub.summary or ''
            }
            for sub in submissions
        ]
        
    finally:
        db.close()


def update_submission_status(submission_id: str, status: str) -> Dict[str, str]:
    """Update submission status in PostgreSQL database."""
    db = SessionLocal()
    try:
        submission = db.query(SubmissionModel).filter(
            SubmissionModel.id == submission_id
        ).first()
        
        if not submission:
            raise ValueError('Submission not found')
        
        submission.status = status
        db.commit()
        
        record_update(f"Submission {submission_id} marked {status}")
        
        return {
            "id": submission.id,
            "title": submission.title,
            "author": submission.author,
            "submitted_at": submission.created_at.isoformat() + 'Z',
            "status": submission.status,
            "mission": submission.submission_type,
            "link": submission.content_url or '',
            "reviewed_at": datetime.utcnow().isoformat() + 'Z'
        }
        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def get_submissions_summary() -> Dict[str, any]:
    """Get summary statistics for community submissions."""
    db = SessionLocal()
    try:
        total_submissions = db.query(func.count(SubmissionModel.id)).scalar()
        
        status_counts = db.query(
            SubmissionModel.status,
            func.count(SubmissionModel.id)
        ).group_by(SubmissionModel.status).all()
        
        type_counts = db.query(
            SubmissionModel.submission_type,
            func.count(SubmissionModel.id)
        ).group_by(SubmissionModel.submission_type).all()
        
        # Get recent submissions (last 30 days)
        from datetime import timedelta
        recent_date = datetime.utcnow() - timedelta(days=30)
        recent_submissions = db.query(func.count(SubmissionModel.id)).filter(
            SubmissionModel.created_at >= recent_date
        ).scalar()
        
        return {
            "total_submissions": total_submissions or 0,
            "recent_submissions_30d": recent_submissions or 0,
            "status_breakdown": {
                status: count for status, count in status_counts
            },
            "type_breakdown": {
                sub_type: count for sub_type, count in type_counts
            },
            "pending_review": db.query(func.count(SubmissionModel.id)).filter(
                SubmissionModel.status == 'pending'
            ).scalar() or 0,
            "approval_rate": 0.85,  # Mock for now - would calculate from historical data
            "generated_at": datetime.utcnow().isoformat()
        }
        
    finally:
        db.close()
