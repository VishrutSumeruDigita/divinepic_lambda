from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
import boto3
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from elasticsearch import Elasticsearch
from typing import List

# ─── Setup logging for Lambda ──────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Only use StreamHandler for Lambda (no file logging)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

# ─── Load AWS credentials & config from .env ────────────────────────────────────
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "divinepic-test")

# ─── Local folder to store uploaded files temporarily ─────────────────────────
TEMP_UPLOAD_DIR = "/tmp/uploads"

# ─── Elasticsearch setup (Lambda-compatible) ────────────────────────────────────
ES_HOSTS = [
    "http://13.202.43.6:9200",   # Remote ES host only (no localhost in Lambda)
]

INDEX_NAME = "face_embeddings"

# ─── Initialize AWS S3 client ──────────────────────────────────────────────────
s3_client = boto3.client("s3", region_name=AWS_REGION)

# ─── Initialize Elasticsearch client ──────────────────────────────────────────
es_clients = []
for host in ES_HOSTS:
    client = Elasticsearch([host], verify_certs=False, timeout=30)
    try:
        info = client.info()
        version = info.get("version", {}).get("number", "<unknown>")
        logger.info(f"✅ Connected to Elasticsearch at {host} — version {version}")
        es_clients.append((client, host))
    except Exception as e:
        logger.error(f"⚠️ Could not connect to Elasticsearch at {host}: {e}")

# ─── FastAPI app instance ───────────────────────────────────────────────────────
app = FastAPI(title="DivinePic Face Detection API", version="1.0.0")

# ─── Ensure temporary upload directory exists ─────────────────────────────────
if not os.path.isdir(TEMP_UPLOAD_DIR):
    os.makedirs(TEMP_UPLOAD_DIR)

# ─── Extract date from filename ────────────────────────────────────────────────
def extract_date_from_filename(filename: str) -> str:
    """Extract date from filename and return formatted date string (DD_MON_YYYY)."""
    try:
        parts = filename.split("_")
        ts_ms = int(parts[1])
        dt = time.strftime("%d_%b_%Y", time.gmtime(ts_ms / 1000.0))
        return dt.upper()
    except (IndexError, ValueError):
        # Fallback to current date if filename doesn't follow expected format
        return time.strftime("%d_%b_%Y", time.gmtime()).upper()

# ─── Upload image to S3 ────────────────────────────────────────────────────────
def upload_image_to_s3(local_path: str) -> str:
    """Upload image to S3 and return public URL."""
    orig_name = Path(local_path).name
    date_str = extract_date_from_filename(orig_name)
    unique_id = uuid.uuid4().hex[:6]
    new_name = f"{date_str}_{unique_id}_{orig_name}"
    s3_key = new_name

    with open(local_path, "rb") as f:
        raw = f.read()

    try:
        s3_client.put_object(
            Bucket=S3_BUCKET, Key=s3_key, Body=raw, ContentType=f"image/{Path(orig_name).suffix.lstrip('.')}"
        )
    except Exception as e:
        logger.error(f"⚠️ Failed to upload '{local_path}' to S3: {e}")
        raise

    public_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    return public_url

# ─── Initialize face detection model (global to avoid reloading) ──────────────
face_app = None

def get_face_model():
    """Get or initialize the face detection model (singleton pattern)."""
    global face_app
    if face_app is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {DEVICE}")

        face_app = FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"] if DEVICE == "cpu" else ["CUDAExecutionProvider"])
        face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_thresh=0.35, det_size=(640, 640))

        logger.info("✅ antelopev2 model loaded successfully")
    return face_app

# ─── Process images and generate embeddings (Lambda-optimized) ────────────────
def process_images_and_generate_embeddings(image_paths: List[str]):
    """Process images to generate face embeddings and index them into Elasticsearch."""
    face_model = get_face_model()  # Get the model instance
    results = []
    
    for image_path in image_paths:
        try:
            # Upload image to S3
            s3_url = upload_image_to_s3(image_path)
            
            # Read image and run face detection
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                logger.warning(f"⚠️ Could not read image '{image_path}'. Skipping...")
                continue
                
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            faces = face_model.get(img_rgb)
            num_faces = len(faces)
            
            if not faces:
                logger.info(f"ℹ️  No faces detected in '{image_path}'. Skipping...")
                results.append({"image": Path(image_path).name, "faces": 0, "status": "no_faces"})
                continue
            
            # Log faces and index to Elasticsearch
            for idx, face in enumerate(faces):
                emb_vec = face.normed_embedding
                box_coords = face.bbox.tolist()

                # Create the Elasticsearch document for this face
                doc = {
                    "image_name": s3_url,
                    "embeds": emb_vec.tolist(),
                    "box": box_coords
                }

                # Index the face into Elasticsearch
                for client, host in es_clients:
                    try:
                        face_id = f"{Path(image_path).stem}_face_{idx+1}_{uuid.uuid4().hex[:8]}"
                        client.index(index=INDEX_NAME, id=face_id, document=doc)
                        logger.info(f"✅ Indexed face {idx+1} from '{image_path}' into Elasticsearch ({host})")
                    except Exception as e:
                        logger.error(f"❌ Failed to index face {idx+1} from '{image_path}' into ES ({host}): {e}")
            
            results.append({"image": Path(image_path).name, "faces": num_faces, "status": "processed", "s3_url": s3_url})
            
        except Exception as e:
            logger.error(f"⚠️ Error processing '{image_path}': {e}")
            results.append({"image": Path(image_path).name, "status": "error", "error": str(e)})
        finally:
            # Clean up temporary file
            try:
                os.remove(image_path)
            except:
                pass
    
    return results

# ─── Health check endpoint ─────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "🎉 DivinePic Face Detection API is LIVE!",
        "status": "Active",
        "version": "1.0.0",
        "environment": {
            "aws_region": os.getenv('AWS_REGION', 'not-set'),
            "s3_bucket": os.getenv('S3_BUCKET_NAME', 'not-set'),
            "es_connections": len(es_clients)
        },
        "endpoints": {
            "POST /upload-images/": "Upload multiple images for face detection"
        }
    }

# ─── Bulk image upload endpoint (Lambda-optimized) ─────────────────────────────
@app.post("/upload-images/")
async def upload_images(files: List[UploadFile] = File(...)):
    """Endpoint to upload multiple images and process them synchronously for Lambda."""
    logger.info(f"⏳ Starting upload and processing of {len(files)} images...")

    # Temporarily store the images in the local directory
    image_paths = []
    for uploaded_file in files:
        # Sanitize filename
        safe_filename = "".join(c for c in uploaded_file.filename if c.isalnum() or c in "._-")
        if not safe_filename:
            safe_filename = f"image_{uuid.uuid4().hex[:8]}.jpg"
            
        image_path = os.path.join(TEMP_UPLOAD_DIR, safe_filename)
        
        try:
            with open(image_path, "wb") as buffer:
                content = await uploaded_file.read()
                buffer.write(content)
            image_paths.append(image_path)
            logger.info(f"✅ Saved {uploaded_file.filename} to {image_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save {uploaded_file.filename}: {e}")
    
    if not image_paths:
        return JSONResponse(
            status_code=400,
            content={"error": "No valid images were uploaded"}
        )
    
    logger.info(f"✅ {len(image_paths)} images uploaded successfully to backend.")
    
    # Process images synchronously (no background tasks for Lambda reliability)
    try:
        results = process_images_and_generate_embeddings(image_paths)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Images uploaded and processed successfully!",
                "total_images": len(files),
                "processed_images": len(results),
                "results": results
            }
        )
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )

# ─── Main function to run the FastAPI server ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
