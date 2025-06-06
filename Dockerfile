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
COPY requirements-cpu.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Create model cache directory (models will be downloaded on first use)
RUN mkdir -p /tmp/insightface_models

# Set function handler
CMD ["lambda_handler.lambda_handler"] 