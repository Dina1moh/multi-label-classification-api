from pydantic import BaseModel
from typing import List

class Response(BaseModel):
    image_name: str
    classes: List[str]
    confidence: List[float]
    
    