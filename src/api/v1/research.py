from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.src.services.research_service import ResearchService, get_research_service
from backend.src.services.peer_review_service import PeerReviewService
from functools import lru_cache


@lru_cache
def get_peer_review_service() -> PeerReviewService:
    return PeerReviewService()


router = APIRouter(prefix="/api/v1/research", tags=["research"])


@router.get("/history")
async def get_history(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(720, ge=1, le=2000),
    service: ResearchService = Depends(get_research_service),
):
    data = service.query_history(start, end, limit)
    return {"index_name": "geri_v1.0", "count": len(data), "data": [record.__dict__ for record in data]}


@router.get("/methodology")
async def get_methodology(service: ResearchService = Depends(get_research_service)):
    documents = service.methodology_documents()
    if not documents:
        raise HTTPException(status_code=404, detail="No methodology documents found")
    return {
        "current_version": "geri_v1.0",
        "documents": documents,
        "citation": "RiskSX Intelligence System (2025). Global Economic Resilience Index (GERI v1.0).",
    }


@router.get("/reviews")
async def list_reviews(service: PeerReviewService = Depends(get_peer_review_service), limit: int = Query(20, ge=1, le=100)):
    reviews = service.list_reviews(limit)
    return {"count": len(reviews), "reviews": [review.__dict__ for review in reviews]}


@router.post("/reviews")
async def submit_review(payload: dict, service: PeerReviewService = Depends(get_peer_review_service)):
    name = payload.get("name")
    decision = payload.get("decision")
    if not name or not decision:
        raise HTTPException(status_code=400, detail="name and decision are required")
    review = service.submit(name, payload.get("email"), decision, payload.get("comments"))
    return review.__dict__
