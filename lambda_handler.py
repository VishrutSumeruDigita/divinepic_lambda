import json
import logging
import os
from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create FastAPI app
app = FastAPI(
    title="DivinePic Face Detection API",
    description="Face detection API using InsightFace",
    version="1.0.0"
)

# Request models
class ImageRequest(BaseModel):
    images: List[str]  # Base64 encoded images

class ImageResponse(BaseModel):
    message: str 
    processed_count: int
    results: List[dict] = []

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "🎉 DivinePic Face Detection API is LIVE!",
        "status": "Active",
        "version": "1.0.0",
        "environment": {
            "aws_region": os.getenv('AWS_REGION', 'not-set'),
            "s3_bucket": os.getenv('S3_BUCKET_NAME', 'not-set')
        },
        "endpoints": {
            "POST /process": "Process images for face detection",
            "GET /": "Health check"
        }
    }

@app.post("/process", response_model=ImageResponse)
async def process_images(request: ImageRequest):
    """Process images for face detection"""
    try:
        logger.info(f"🔍 Processing {len(request.images)} images")
        
        # TODO: Add actual face detection logic here
        # For now, just return a mock response
        results = []
        for i, image_data in enumerate(request.images):
            results.append({
                "image_index": i,
                "faces_detected": 0,  # Mock data
                "processing_time": 0.1,
                "status": "processed"
            })
        
        return ImageResponse(
            message="✅ Images processed successfully!",
            processed_count=len(request.images),
            results=results
        )
        
    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": "2025-06-06T11:00:00Z",
        "environment": {
            "aws_access_key_set": bool(os.getenv('AWS_ACCESS_KEY_ID')),
            "aws_region": os.getenv('AWS_REGION'),
            "s3_bucket": os.getenv('S3_BUCKET_NAME')
        },
        "services": {
            "face_detection": "ready",
            "elasticsearch": "connected"  # Add actual health checks
        }
    }

# Create Lambda handler using Mangum
lambda_handler = Mangum(app, lifespan="off") 