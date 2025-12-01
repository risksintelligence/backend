from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import logging
from slugify import slugify

from app.db import get_db
from app.models import BlogPost, BlogCategory, BlogEngagement

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/blog", tags=["blog"])

class BlogPostCreate(BaseModel):
    title: str
    excerpt: str
    content: str
    author_name: str
    author_title: str
    author_linkedin: Optional[str] = None
    category: str
    tags: List[str] = []
    meta_description: Optional[str] = None
    featured_image_url: Optional[str] = None
    is_featured: bool = False

class BlogPostUpdate(BaseModel):
    title: Optional[str] = None
    excerpt: Optional[str] = None
    content: Optional[str] = None
    author_name: Optional[str] = None
    author_title: Optional[str] = None
    author_linkedin: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    meta_description: Optional[str] = None
    featured_image_url: Optional[str] = None
    is_featured: Optional[bool] = None

class BlogPostResponse(BaseModel):
    id: int
    title: str
    slug: str
    excerpt: str
    content: str
    author_name: str
    author_title: str
    author_linkedin: Optional[str]
    published_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    is_published: bool
    is_featured: bool
    category: str
    tags: List[str]
    meta_description: Optional[str]
    featured_image_url: Optional[str]
    reading_time_minutes: Optional[int]
    view_count: int

    class Config:
        from_attributes = True

class BlogPostSummary(BaseModel):
    id: int
    title: str
    slug: str
    excerpt: str
    author_name: str
    published_at: Optional[datetime]
    category: str
    tags: List[str]
    reading_time_minutes: Optional[int]
    view_count: int
    is_featured: bool

    class Config:
        from_attributes = True

class BlogCategoryResponse(BaseModel):
    id: int
    name: str
    slug: str
    description: Optional[str]
    color: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

def calculate_reading_time(content: str) -> int:
    """Calculate estimated reading time in minutes (200 WPM average)"""
    word_count = len(content.split())
    return max(1, round(word_count / 200))

@router.get("/posts", response_model=List[BlogPostSummary])
async def get_blog_posts(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, le=100),
    category: Optional[str] = None,
    featured_only: bool = False,
    published_only: bool = True,
    db: Session = Depends(get_db)
):
    """Get blog posts with pagination and filtering."""
    query = db.query(BlogPost)
    
    if published_only:
        query = query.filter(BlogPost.is_published == True)
    
    if category:
        query = query.filter(BlogPost.category == category)
        
    if featured_only:
        query = query.filter(BlogPost.is_featured == True)
    
    posts = query.order_by(BlogPost.published_at.desc()).offset(skip).limit(limit).all()
    return posts

@router.get("/posts/{slug}", response_model=BlogPostResponse)
async def get_blog_post(slug: str, db: Session = Depends(get_db)):
    """Get a single blog post by slug."""
    post = db.query(BlogPost).filter(BlogPost.slug == slug).first()
    
    if not post:
        raise HTTPException(status_code=404, detail="Blog post not found")
    
    # Increment view count
    post.view_count = (post.view_count or 0) + 1
    db.commit()
    
    return post

@router.post("/posts", response_model=BlogPostResponse)
async def create_blog_post(post_data: BlogPostCreate, db: Session = Depends(get_db)):
    """Create a new blog post."""
    try:
        # Generate slug from title
        slug = slugify(post_data.title)
        
        # Ensure slug is unique
        existing_post = db.query(BlogPost).filter(BlogPost.slug == slug).first()
        if existing_post:
            slug = f"{slug}-{int(datetime.utcnow().timestamp())}"
        
        # Calculate reading time
        reading_time = calculate_reading_time(post_data.content)
        
        # Create post
        db_post = BlogPost(
            title=post_data.title,
            slug=slug,
            excerpt=post_data.excerpt,
            content=post_data.content,
            author_name=post_data.author_name,
            author_title=post_data.author_title,
            author_linkedin=post_data.author_linkedin,
            category=post_data.category,
            tags=post_data.tags,
            meta_description=post_data.meta_description or post_data.excerpt[:300],
            featured_image_url=post_data.featured_image_url,
            reading_time_minutes=reading_time,
            is_featured=post_data.is_featured,
            is_published=False,  # Draft by default
            view_count=0
        )
        
        db.add(db_post)
        db.commit()
        db.refresh(db_post)
        
        logger.info(f"Created blog post: {post_data.title}")
        return db_post
        
    except Exception as e:
        logger.error(f"Error creating blog post: {e}")
        raise HTTPException(status_code=500, detail="Failed to create blog post")

@router.put("/posts/{slug}", response_model=BlogPostResponse)
async def update_blog_post(
    slug: str, 
    post_data: BlogPostUpdate, 
    db: Session = Depends(get_db)
):
    """Update a blog post."""
    try:
        post = db.query(BlogPost).filter(BlogPost.slug == slug).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Update fields
        update_data = post_data.dict(exclude_unset=True)
        
        # Recalculate reading time if content changed
        if "content" in update_data:
            update_data["reading_time_minutes"] = calculate_reading_time(update_data["content"])
        
        # Update slug if title changed
        if "title" in update_data and update_data["title"] != post.title:
            new_slug = slugify(update_data["title"])
            if new_slug != post.slug:
                existing_post = db.query(BlogPost).filter(BlogPost.slug == new_slug).first()
                if not existing_post:
                    update_data["slug"] = new_slug
        
        for field, value in update_data.items():
            setattr(post, field, value)
        
        post.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(post)
        
        logger.info(f"Updated blog post: {slug}")
        return post
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating blog post: {e}")
        raise HTTPException(status_code=500, detail="Failed to update blog post")

@router.post("/posts/{slug}/publish")
async def publish_blog_post(slug: str, db: Session = Depends(get_db)):
    """Publish a blog post."""
    try:
        post = db.query(BlogPost).filter(BlogPost.slug == slug).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        post.is_published = True
        post.published_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Published blog post: {slug}")
        return {"status": "success", "message": "Blog post published"}
        
    except Exception as e:
        logger.error(f"Error publishing blog post: {e}")
        raise HTTPException(status_code=500, detail="Failed to publish blog post")

@router.delete("/posts/{slug}")
async def delete_blog_post(slug: str, db: Session = Depends(get_db)):
    """Delete a blog post."""
    try:
        post = db.query(BlogPost).filter(BlogPost.slug == slug).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        db.delete(post)
        db.commit()
        
        logger.info(f"Deleted blog post: {slug}")
        return {"status": "success", "message": "Blog post deleted"}
        
    except Exception as e:
        logger.error(f"Error deleting blog post: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete blog post")

@router.get("/categories", response_model=List[BlogCategoryResponse])
async def get_categories(db: Session = Depends(get_db)):
    """Get all blog categories."""
    categories = db.query(BlogCategory).order_by(BlogCategory.name).all()
    return categories

@router.post("/track-engagement")
async def track_content_engagement(
    post_slug: str,
    action: str,
    session_id: Optional[str] = None,
    metadata: Dict[str, Any] = {},
    db: Session = Depends(get_db)
):
    """Track content engagement events."""
    try:
        engagement = BlogEngagement(
            post_slug=post_slug,
            action=action,
            session_id=session_id,
            extra_data=metadata
        )
        
        db.add(engagement)
        db.commit()
        
        return {"status": "success", "message": "Engagement tracked"}
        
    except Exception as e:
        logger.error(f"Error tracking engagement: {e}")
        raise HTTPException(status_code=500, detail="Failed to track engagement")

@router.get("/analytics")
async def get_blog_analytics(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get blog analytics for admin dashboard."""
    try:
        from datetime import timedelta
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Post statistics
        total_posts = db.query(BlogPost).count()
        published_posts = db.query(BlogPost).filter(BlogPost.is_published == True).count()
        
        # Engagement metrics
        engagements = db.query(BlogEngagement).filter(
            BlogEngagement.timestamp >= start_date
        ).all()
        
        # Top posts by views
        top_posts = db.query(BlogPost).filter(
            BlogPost.is_published == True
        ).order_by(BlogPost.view_count.desc()).limit(5).all()
        
        return {
            "summary": {
                "total_posts": total_posts,
                "published_posts": published_posts,
                "draft_posts": total_posts - published_posts,
                "total_views": sum(post.view_count or 0 for post in db.query(BlogPost).all()),
                "total_engagements": len(engagements)
            },
            "top_posts": [
                {
                    "title": post.title,
                    "slug": post.slug,
                    "view_count": post.view_count,
                    "published_at": post.published_at.isoformat() if post.published_at else None
                }
                for post in top_posts
            ],
            "engagement_by_action": {
                action: len([e for e in engagements if e.action == action])
                for action in set(e.action for e in engagements)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting blog analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get blog analytics")