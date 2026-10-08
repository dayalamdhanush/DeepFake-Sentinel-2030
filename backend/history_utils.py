import os
import time
from typing import Dict, Any
from sqlalchemy.orm import Session
from backend.api.history import HistoryEntry

def log_history(db: Session, filename: str, media_type: str, result: Dict[str, Any], model_name: str, processing_time_ms: int, modality_results: Dict[str, Any] = None):
    """Insert a history record.
    * result must contain keys: prediction (label), confidence, risk.
    * modality_results is optional dict of per-modality results for multimodal.
    """
    entry = HistoryEntry(
        filename=filename,
        media_type=media_type,
        prediction=result.get('label') or result.get('prediction') or 'unknown',
        confidence=result.get('confidence', 0.0),
        risk=result.get('risk', 'unknown'),
        model_name=model_name,
        processing_time_ms=processing_time_ms,
        timestamp=time.time(),
        modality_results=modality_results,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry.id
