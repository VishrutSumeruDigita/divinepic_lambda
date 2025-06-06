FROM public.ecr.aws/lambda/python:3.10
# Copy function code
COPY ./app ${LAMBDA_TASK_ROOT}
# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY requirements.txt .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}" -U --no-cache-dir

# Create a script to cache InsightFace models
RUN echo 'import os' > /tmp/cache_models.py && \
    echo 'os.makedirs("/tmp/insightface_models", exist_ok=True)' >> /tmp/cache_models.py && \
    echo 'from insightface.app import FaceAnalysis' >> /tmp/cache_models.py && \
    echo 'try:' >> /tmp/cache_models.py && \
    echo '    app = FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])' >> /tmp/cache_models.py && \
    echo '    app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))' >> /tmp/cache_models.py && \
    echo '    print("✅ InsightFace models cached successfully")' >> /tmp/cache_models.py && \
    echo 'except Exception as e:' >> /tmp/cache_models.py && \
    echo '    print(f"⚠️ Model caching failed: {e}")' >> /tmp/cache_models.py

# Run the script to cache models
RUN python /tmp/cache_models.py && rm /tmp/cache_models.py

# Set the CMD to your handler
CMD ["app.handler"]
