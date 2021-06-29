from pydantic import BaseModel
from typing import Dict, Optional, Any, List


class ModelPayload(BaseModel):
    inputs: Optional[Any] = None
    params: Optional[Dict]= {}

class CompletionPayload(BaseModel):
    context: str
    temp: Optional[float] = 1.0
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    length: Optional[int] = 256

class CompletionResponse(BaseModel):
    context: str
    completion: str
    time: float

class QueueResponse(BaseModel):
    qid: int

class QueueRequest(BaseModel):
    qid: int
