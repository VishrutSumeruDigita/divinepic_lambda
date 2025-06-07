# Use AWS Lambda Python 3.11 runtime
FROM public.ecr.aws/lambda/python:3.11

# Set environment variables to optimize ML libraries for Lambda
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/app/.matplotlib
ENV NUMBA_CACHE_DIR=/tmp
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
# Fix InsightFace paths
ENV INSIGHTFACE_ROOT=/app/.insightface

# Install system dependencies needed for OpenCV and other libraries
RUN yum update -y && \
    yum install -y \
    gcc \
    gcc-c++ \
    make \
    cmake \
    wget \
    unzip \
    tar \
    gzip \
    libgomp \
    libstdc++ \
    mesa-libGL \
    && yum clean all

# Copy requirements first for better caching
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download and setup AntelopeV2 model files
RUN cd /tmp && \
    wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip && \
    if [ ! -f "antelopev2.zip" ]; then echo "Failed to download model"; exit 1; fi && \
    unzip -q antelopev2.zip -d antelopev2_files && \
    if [ ! -d "antelopev2_files" ]; then echo "Failed to extract model"; exit 1; fi && \
    mkdir -p /app/models/antelopev2/detection && \
    cp antelopev2_files/antelopev2/scrfd_10g_bnkps.onnx /app/models/antelopev2/detection/ && \
    cp antelopev2_files/antelopev2/glintr100.onnx /app/models/antelopev2/ && \
    if [ ! -f "/app/models/antelopev2/detection/scrfd_10g_bnkps.onnx" ] || [ ! -f "/app/models/antelopev2/glintr100.onnx" ]; then \
        echo "Failed to copy model files"; exit 1; \
    fi && \
    rm -rf /tmp/antelopev2*

# Copy application code
COPY app.py ${LAMBDA_TASK_ROOT}/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Create necessary directories
RUN mkdir -p /tmp/uploads && \
    mkdir -p /app/.insightface && \
    mkdir -p /app/.matplotlib

# Set the CMD to your handler
CMD ["lambda_handler.lambda_handler"] 