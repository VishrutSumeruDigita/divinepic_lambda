import os
import logging
from mangum import Mangum
from app import app

# Configure logging for Lambda
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the Mangum adapter
mangum_handler = Mangum(app, lifespan="off")

# Lambda handler with error handling
def lambda_handler(event, context):
    """
    AWS Lambda handler function that wraps the FastAPI app.
    """
    try:
        logger.info(f"Received event: {event.get('httpMethod', 'UNKNOWN')} {event.get('path', 'UNKNOWN')}")
        return mangum_handler(event, context)
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
            },
            "body": '{"error": "Internal server error"}'
        } 