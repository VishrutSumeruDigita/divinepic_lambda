#!/bin/bash

# AWS Lambda Deployment Script for Face Detection API
# Usage: ./deploy.sh

set -e

# Configuration
AWS_REGION="ap-south-1"
AWS_ACCOUNT_ID="756276770091"
ECR_REPOSITORY="divinepic-cpu-lambda"
LAMBDA_FUNCTION_NAME="divinepic-cpu-lambda"
IMAGE_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting AWS Lambda deployment...${NC}"

# Step 1: Login to ECR
echo -e "${YELLOW}📝 Step 1: Logging in to Amazon ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Step 2: Build Docker image
echo -e "${YELLOW}🔨 Step 2: Building Docker image...${NC}"
docker build --platform linux/amd64 -t $ECR_REPOSITORY:$IMAGE_TAG .

# Step 3: Tag image for ECR
echo -e "${YELLOW}🏷️  Step 3: Tagging image for ECR...${NC}"
docker tag $ECR_REPOSITORY:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

# Step 4: Push image to ECR
echo -e "${YELLOW}⬆️  Step 4: Pushing image to ECR...${NC}"
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

# Step 5: Update Lambda function
echo -e "${YELLOW}⚡ Step 5: Updating Lambda function...${NC}"
aws lambda update-function-code \
    --region $AWS_REGION \
    --function-name $LAMBDA_FUNCTION_NAME \
    --image-uri $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

# Step 6: Wait for update to complete
echo -e "${YELLOW}⏳ Step 6: Waiting for Lambda function update to complete...${NC}"
aws lambda wait function-updated --region $AWS_REGION --function-name $LAMBDA_FUNCTION_NAME

# Step 7: Get function URL
echo -e "${YELLOW}🔗 Step 7: Getting function URL...${NC}"
FUNCTION_URL=$(aws lambda get-function-url-config --region $AWS_REGION --function-name $LAMBDA_FUNCTION_NAME --query 'FunctionUrl' --output text 2>/dev/null || echo "No function URL configured")

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${BLUE}📊 Deployment Summary:${NC}"
echo -e "  • ECR Repository: ${ECR_REPOSITORY}"
echo -e "  • Image URI: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG"
echo -e "  • Lambda Function: ${LAMBDA_FUNCTION_NAME}"
echo -e "  • Function URL: ${FUNCTION_URL}"
echo ""
echo -e "${BLUE}🧪 Test your API:${NC}"
echo -e "  • Health Check: curl ${FUNCTION_URL}health"
echo -e "  • ES Test: curl ${FUNCTION_URL}test-es"
echo -e "  • Upload Images: curl -X POST ${FUNCTION_URL}upload-images/ -F 'files=@image.jpg'" 