import os
import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

router = APIRouter()

# SQLite setup (synchronous)
DATABASE_URL = "sqlite:///./deepfake_history.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class HistoryEntry(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    media_type = Column(String, nullable=False)  # image, video, audio, multimodal
    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    risk = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    processing_time_ms = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    modality_results = Column(JSON, nullable=True)  # store dict of per-modality results

# Create tables if not exist
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/history", response_model=List[dict])
def get_history(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100), media_type: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(HistoryEntry).order_by(HistoryEntry.timestamp.desc())
    if media_type:
        query = query.filter(HistoryEntry.media_type == media_type)
    total = query.count()
    entries = query.offset((page - 1) * page_size).limit(page_size).all()
    return [
        {
            "id": e.id,
            "filename": e.filename,
            "media_type": e.media_type,
            "prediction": e.prediction,
            "confidence": e.confidence,
            "risk": e.risk,
            "model_name": e.model_name,
            "processing_time_ms": e.processing_time_ms,
            "timestamp": e.timestamp.isoformat(),
            "modality_results": e.modality_results,
        }
        for e in entries
    ]

@router.get("/history/{entry_id}", response_model=dict)
def get_history_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(HistoryEntry).filter(HistoryEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="History entry not found")
    return {
        "id": entry.id,
        "filename": entry.filename,
        "media_type": entry.media_type,
        "prediction": entry.prediction,
        "confidence": entry.confidence,
        "risk": entry.risk,
        "model_name": entry.model_name,
        "processing_time_ms": entry.processing_time_ms,
        "timestamp": entry.timestamp.isoformat(),
        "modality_results": entry.modality_results,
    }

@router.delete("/history/{entry_id}")
def delete_history_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(HistoryEntry).filter(HistoryEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="History entry not found")
    db.delete(entry)
    db.commit()
    return {"detail": f"History entry {entry_id} deleted"}
