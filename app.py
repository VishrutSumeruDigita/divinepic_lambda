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

# Only use StreamHandler for Lambda
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

# ─── Load AWS credentials & config from .env ────────────────────────────────────
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "divinepic-test")
S3_UPLOAD_PATH = os.getenv("S3_UPLOAD_PATH", "upload_with_embed")

# ─── Local folder to store uploaded files temporarily ─────────────────────────
TEMP_UPLOAD_DIR = "/tmp/uploads"

# ─── Elasticsearch setup ────────────────────────────────────────────────────────
ES_HOSTS = [
    os.getenv("ES_HOSTS1", "http://13.202.43.6:9200"),
    os.getenv("ES_HOSTS2", "http://13.202.43.6:9200")  # Add second host if needed
]

INDEX_NAME = os.getenv("INDEX_NAME", "face_embeddings")

# ─── Initialize AWS S3 client ──────────────────────────────────────────────────
s3_client = boto3.client("s3", region_name=AWS_REGION)

# ─── Initialize Elasticsearch clients (will connect when first used) ──────────────
es_clients = []
for host in ES_HOSTS:
    client = Elasticsearch([host], verify_certs=False)
    es_clients.append((client, host))

# ─── FastAPI app instance ───────────────────────────────────────────────────────
app = FastAPI()

# ─── Ensure temporary upload directory exists ─────────────────────────────────
if not os.path.isdir(TEMP_UPLOAD_DIR):
    os.makedirs(TEMP_UPLOAD_DIR)

# ─── Create Elasticsearch index if needed ─────────────────────────────────────
def create_index_if_not_exists():
    """Create the index (if it doesn't exist) on all ES clusters with the same mapping."""
    mapping_body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "image_name": { "type": "keyword" },
                "embeds": {
                    "type": "dense_vector",
                    "dims": 512,
                    "index": True,
                    "similarity": "cosine"
                },
                "box": {
                    "type": "dense_vector",
                    "dims": 4
                }
            }
        }
    }

    for client, host in es_clients:
        try:
            if client.indices.exists(index=INDEX_NAME):
                logger.info(f"ℹ️  Index '{INDEX_NAME}' already exists on {host}")
            else:
                logger.info(f"🚀 Creating index '{INDEX_NAME}' on {host}")
                client.indices.create(index=INDEX_NAME, body=mapping_body)
                logger.info(f"✅ Index '{INDEX_NAME}' created successfully on {host}")
        except Exception as e:
            logger.error(f"❌ Failed to create index on {host}: {e}")

# ─── Extract date from filename ────────────────────────────────────────────────
def extract_date_from_filename(filename: str) -> str:
    """Extract date from filename and return formatted date string (DD_MON_YYYY)."""
    try:
        parts = filename.split("_")
        ts_ms = int(parts[1])
        dt = time.strftime("%d_%b_%Y", time.gmtime(ts_ms / 1000.0))
        return dt.upper()
    except (IndexError, ValueError):
        return time.strftime("%d_%b_%Y", time.gmtime()).upper()

# ─── Upload image to S3 ────────────────────────────────────────────────────────
def upload_image_to_s3(local_path: str) -> str:
    """Upload image to S3 and return public URL."""
    filename = Path(local_path).name
    file_ext = Path(filename).suffix.lower()
    
    if file_ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
        raise ValueError(f"Unsupported file extension: {file_ext}")

    unique_key = f"{uuid.uuid4().hex}_{filename}"
    s3_key = f"{S3_UPLOAD_PATH.rstrip('/')}/{unique_key}"

    with open(local_path, "rb") as f:
        data = f.read()

    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=data,
            ContentType=f"image/{file_ext.lstrip('.')}"
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

        face_app = FaceAnalysis(
            name="antelopev2",
            root="/app/models",
            providers=["CPUExecutionProvider"] if DEVICE == "cpu" else ["CUDAExecutionProvider"]
        )
        face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_thresh=0.35, det_size=(640, 640))

        logger.info("✅ antelopev2 model loaded successfully")
    return face_app

# ─── Process images in the background (generate embeddings and index to ES) ────
def process_images_and_generate_embeddings(image_paths: List[str]):
    """Process images to generate face embeddings and index them into Elasticsearch."""
    face_model = get_face_model()  # Get the model instance
    
    # Ensure index exists
    create_index_if_not_exists()
    
    processed_count = 0
    total_faces = 0
    
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
                logger.info(f"ℹ️  No faces detected in '{image_path}'. Still uploaded to S3 → URL: {s3_url}")
                processed_count += 1
                continue
            
            # Log faces and index to Elasticsearch
            for idx, face in enumerate(faces):
                emb_vec = face.normed_embedding
                box_coords = face.bbox.tolist()

                face_id = f"{Path(image_path).stem}_face_{idx+1}_{uuid.uuid4().hex[:8]}"
                doc = {
                    "image_name": s3_url,
                    "embeds": emb_vec.tolist(),
                    "box": box_coords
                }

                # Index the face into all Elasticsearch clusters
                for client, host in es_clients:
                    try:
                        client.index(index=INDEX_NAME, id=face_id, document=doc)
                        logger.info(f"✅ Indexed face {idx+1} from '{Path(image_path).name}' into ES ({host})")
                    except Exception as e:
                        logger.error(f"❌ Failed to index face {idx+1} from '{Path(image_path).name}' into ES ({host}): {e}")
            
            total_faces += num_faces
            processed_count += 1
            logger.info(f"✅ Processed '{Path(image_path).name}' → faces: {num_faces}, S3 URL: {s3_url}")
            
        except Exception as e:
            logger.error(f"⚠️ Error processing '{image_path}': {e}")
        finally:
            # Clean up temporary file
            try:
                os.remove(image_path)
            except:
                pass
    
    logger.info(f"🏁 Bulk processing complete: {processed_count} images processed, {total_faces} faces indexed")

# ─── Bulk image upload endpoint ─────────────────────────────────────────────────
@app.post("/upload-images/")
async def upload_images(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Endpoint to upload multiple images and process them in the background."""
    logger.info(f"⏳ Starting bulk image upload for {len(files)} files...")

    if not files:
        return JSONResponse(
            status_code=400,
            content={"error": "No files provided"}
        )

    # Validate file extensions
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    invalid_files = []
    
    # Temporarily store the images in the local directory
    image_paths = []
    for uploaded_file in files:
        file_ext = Path(uploaded_file.filename).suffix.lower()
        if file_ext not in valid_exts:
            invalid_files.append(uploaded_file.filename)
            continue
            
        image_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_file.filename)
        try:
            with open(image_path, "wb") as buffer:
                content = await uploaded_file.read()
                buffer.write(content)
            image_paths.append(image_path)
        except Exception as e:
            logger.error(f"Failed to save uploaded file {uploaded_file.filename}: {e}")
            invalid_files.append(uploaded_file.filename)
    
    if invalid_files:
        logger.warning(f"⚠️ Skipped {len(invalid_files)} invalid files: {invalid_files}")
    
    if not image_paths:
        return JSONResponse(
            status_code=400,
            content={"error": "No valid image files provided", "invalid_files": invalid_files}
        )
    
    logger.info(f"✅ {len(image_paths)} valid images uploaded successfully to backend.")
    
    # Background task to process the images and generate embeddings
    background_tasks.add_task(process_images_and_generate_embeddings, image_paths)

    response_data = {
        "message": f"Images uploaded successfully! Processing {len(image_paths)} images for face detection and embedding generation.",
        "valid_files": len(image_paths),
        "total_files": len(files)
    }
    
    if invalid_files:
        response_data["invalid_files"] = invalid_files
        response_data["skipped_files"] = len(invalid_files)

    return JSONResponse(status_code=200, content=response_data)

# ─── Health check endpoint ─────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "face-detection-api"}

# ─── Elasticsearch connection test endpoint ─────────────────────────────────────
@app.get("/test-es")
async def test_elasticsearch():
    """Test Elasticsearch connections."""
    results = []
    for client, host in es_clients:
        try:
            info = client.info()
            results.append({"host": host, "status": "connected", "version": info.get("version", {}).get("number", "unknown")})
        except Exception as e:
            results.append({"host": host, "status": "error", "error": str(e)})
    
    return {"elasticsearch_connections": results}

# ─── Main function to run the FastAPI server ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
