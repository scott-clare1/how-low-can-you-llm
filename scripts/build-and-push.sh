#!/bin/bash

TAG="latest"

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

aws ecr get-login-password --region $TF_VAR_region | docker login --username AWS --password-stdin $TF_VAR_aws_account_id.dkr.ecr.$TF_VAR_region.amazonaws.com

docker build -t $TF_VAR_image_name src/llmc

docker tag $TF_VAR_image_name:$TAG $TF_VAR_aws_account_id.dkr.ecr.$TF_VAR_region.amazonaws.com/$TF_VAR_ecr_name:$TAG

docker push $TF_VAR_aws_account_id.dkr.ecr.$TF_VAR_region.amazonaws.com/$TF_VAR_ecr_name:$TAG
