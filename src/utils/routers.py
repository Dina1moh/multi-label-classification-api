from fastapi import APIRouter, HTTPException, UploadFile, Request, File, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from src.utils.config import API_SECRET_KEY
from src.utils.models import Response 
from src.inference import Predict

router = APIRouter()

# API Key dependency (for JSON endpoints if needed)
api_key_header = APIKeyHeader(name="X-API-KEY")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    print(f"Received API key: {api_key}")
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@router.get("/check", tags=["check"])
async def home(api_key: str = Depends(api_key_header)):
    return {"message": "Welcome to Muti-Label Image Classification API"}

# JSON prediction endpoint (optional)
@router.post("/prediction", tags=["prediction"], response_model=Response)
async def get_prediction(file: UploadFile, api_key: str = Depends(api_key_header)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image")
        contents = await file.read()
        prediction  = Predict(contents)
        response_data = {
            
            "image_name": file.filename,
            "classes": prediction["classes"],
            "confidence": prediction["confidence"],
        }
        return Response(**response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")
    
    
    
   
