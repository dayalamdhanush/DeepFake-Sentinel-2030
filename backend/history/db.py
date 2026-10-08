# backend/history/db.py
"""SQLite History Database using SQLAlchemy for DeepFake Sentinel 2030.

Stores prediction logs for successful model inferences.
Does NOT store MODEL_NOT_TRAINED error outputs.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../history.db"))
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String, nullable=False)
    media_type = Column(String, nullable=False)  # image, video, audio, fusion
    label = Column(String, nullable=False)        # REAL, FAKE
    confidence = Column(Float, nullable=False)
    risk = Column(String, nullable=False)         # LOW, MEDIUM, HIGH
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def log_prediction(filename: str, media_type: str, label: str, confidence: float, risk: str) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        record = PredictionRecord(
            filename=filename,
            media_type=media_type,
            label=label.upper(),
            confidence=float(confidence),
            risk=risk.upper(),
            timestamp=datetime.utcnow()
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return {
            "id": record.id,
            "filename": record.filename,
            "media_type": record.media_type,
            "label": record.label,
            "confidence": record.confidence,
            "risk": record.risk,
            "timestamp": record.timestamp.isoformat()
        }
    finally:
        db.close()

def get_history(limit: int = 50) -> List[Dict[str, Any]]:
    db = SessionLocal()
    try:
        records = db.query(PredictionRecord).order_by(PredictionRecord.timestamp.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "filename": r.filename,
                "media_type": r.media_type,
                "label": r.label,
                "confidence": r.confidence,
                "risk": r.risk,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None
            }
            for r in records
        ]
    finally:
        db.close()

def get_history_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    db = SessionLocal()
    try:
        r = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
        if not r:
            return None
        return {
            "id": r.id,
            "filename": r.filename,
            "media_type": r.media_type,
            "label": r.label,
            "confidence": r.confidence,
            "risk": r.risk,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None
        }
    finally:
        db.close()
