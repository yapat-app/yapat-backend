"""
Snippet generation and hybrid caching service
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.snippet import Snippet
from app.models.annotation import Annotation


class SnippetService:
    """Service for snippet operations including hybrid caching"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_feed(
        self,
        dataset_id: Optional[int] = None,
        recording_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Snippet]:
        """Get feed of snippets for annotation, prioritizing unannotated snippets"""
        query = self.db.query(Snippet)
        
        if recording_id:
            query = query.filter(Snippet.recording_id == recording_id)
        elif dataset_id:
            # Filter by dataset through recording relationship
            from app.models.recording import Recording
            query = query.join(Recording).filter(Recording.dataset_id == dataset_id)
        
        # Prioritize unannotated snippets
        query = query.filter(Snippet.is_annotated == False)
        
        snippets = query.offset(skip).limit(limit).all()
        return snippets
    
    def get_similar_snippets(self, snippet_id: int, limit: int = 10) -> List[Snippet]:
        """Get similar snippets using embedding similarity"""
        # Get the reference snippet
        reference = self.db.query(Snippet).filter(Snippet.id == snippet_id).first()
        if not reference or not reference.embedding:
            return []
        
        # TODO: Implement vector similarity search
        # For now, return empty list
        # This would use a vector database or PostgreSQL pgvector extension
        return []
    
    def mark_as_annotated(self, snippet_id: int):
        """Mark a snippet as annotated"""
        snippet = self.db.query(Snippet).filter(Snippet.id == snippet_id).first()
        if snippet:
            snippet.is_annotated = True
            self.db.commit()
    
    def get_annotation_count(self, snippet_id: int) -> int:
        """Get the number of annotations for a snippet"""
        return self.db.query(Annotation).filter(Annotation.snippet_id == snippet_id).count()

