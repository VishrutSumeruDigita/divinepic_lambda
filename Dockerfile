# Lightweight CPU-only Lambda Container
FROM public.ecr.aws/lambda/python:3.9

# Install minimal system dependencies for CPU processing
RUN yum update -y && \
    yum install -y \
    gcc \
    gcc-c++ \
    make \
    cmake \
    && yum clean all && \
    rm -rf /var/cache/yum

# Copy and install Python dependencies (CPU versions)
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment variables (if exists)
COPY .env* ${LAMBDA_TASK_ROOT}/

# Copy the entire app directory
COPY app/ ${LAMBDA_TASK_ROOT}/app/

# Create a Lambda wrapper for the existing FastAPI app
RUN echo 'from mangum import Mangum' > ${LAMBDA_TASK_ROOT}/lambda_handler.py && \
    echo 'from app.app import app' >> ${LAMBDA_TASK_ROOT}/lambda_handler.py && \
    echo '' >> ${LAMBDA_TASK_ROOT}/lambda_handler.py && \
    echo 'lambda_handler = Mangum(app, lifespan="off")' >> ${LAMBDA_TASK_ROOT}/lambda_handler.py

# Create model cache directory (models will be downloaded on first use)
RUN mkdir -p /tmp/insightface_models
RUN mkdir -p /tmp/uploads

# Set function handler
CMD ["lambda_handler.lambda_handler"] 