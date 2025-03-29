#!/bin/bash

IMAGE_NAME="server"
TAG="latest"

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

docker build -t $IMAGE_NAME src/llmc

docker tag $IMAGE_NAME:$TAG $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$TF_VAR_ecr_name:$TAG

docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$TF_VAR_ecr_name:$TAG
