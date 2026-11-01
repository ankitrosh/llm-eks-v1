#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

MODE="${1:---check}"
REGION="us-east-2"
ACCOUNT="657173565477"
PROJECT="qwen3-v1-image-build"
ROLE="llm-qwen3-codebuild-role-v1"
BUCKET="llm-qwen3-buildsrc-${ACCOUNT}-${REGION}"
KEY="source/qwen3-codebuild-source.zip"
SOURCE="test-results/qwen3-codebuild-source.zip"
OUT="infrastructure/codebuild"

if [[ "$MODE" != "--check" && "$MODE" != "--apply" ]]; then
  echo "Usage: $0 [--check|--apply]"
  exit 2
fi

mkdir -p "$OUT"

ACTUAL_ACCOUNT=$(aws sts get-caller-identity \
  --query Account --output text)

if [[ "$ACTUAL_ACCOUNT" != "$ACCOUNT" ]]; then
  echo "ERROR: Unexpected AWS account: $ACTUAL_ACCOUNT"
  exit 1
fi

test -s "$SOURCE" || {
  echo "ERROR: Missing source archive: $SOURCE"
  exit 1
}

export REGION ACCOUNT PROJECT ROLE BUCKET KEY

python3 - <<'PY'
import json
import os
from pathlib import Path

out = Path("infrastructure/codebuild")
out.mkdir(exist_ok=True)

region = os.environ["REGION"]
account = os.environ["ACCOUNT"]
project = os.environ["PROJECT"]
role = os.environ["ROLE"]
bucket = os.environ["BUCKET"]
key = os.environ["KEY"]

project_arn = (
    f"arn:aws:codebuild:{region}:{account}:project/{project}"
)

role_arn = f"arn:aws:iam::{account}:role/{role}"
log_group = f"/aws/codebuild/{project}"
log_arn = (
    f"arn:aws:logs:{region}:{account}:log-group:{log_group}"
)

trust = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "codebuild.amazonaws.com"},
        "Action": "sts:AssumeRole",
        "Condition": {
            "StringEquals": {
                "aws:SourceAccount": account,
                "aws:SourceArn": project_arn
            }
        }
    }]
}

policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ReadSourceBucket",
            "Effect": "Allow",
            "Action": [
                "s3:GetBucketLocation",
                "s3:GetBucketAcl"
            ],
            "Resource": f"arn:aws:s3:::{bucket}"
        },
        {
            "Sid": "ReadSourceObject",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:GetObjectVersion"
            ],
            "Resource": f"arn:aws:s3:::{bucket}/{key}"
        },
        {
            "Sid": "WriteBuildLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": [
                log_arn,
                log_arn + ":*"
            ]
        }
    ]
}

config = {
    "name": project,
    "description": "Qwen3-14B NCCL pipeline V1 image rebuild",
    "source": {
        "type": "S3",
        "location": f"{bucket}/{key}",
        "buildspec": "buildspec.yml"
    },
    "artifacts": {"type": "NO_ARTIFACTS"},
    "environment": {
        "type": "LINUX_CONTAINER",
        "image": "aws/codebuild/standard:8.0",
        "computeType": "BUILD_GENERAL1_MEDIUM",
        "privilegedMode": True,
        "imagePullCredentialsType": "CODEBUILD",
        "environmentVariables": [
            {
                "name": "PUBLISH_TO_ECR",
                "value": "false",
                "type": "PLAINTEXT"
            },
            {
                "name": "IMAGE_TAG",
                "value": "v1-rebuild-test",
                "type": "PLAINTEXT"
            }
        ]
    },
    "serviceRole": role_arn,
    "timeoutInMinutes": 90,
    "cache": {"type": "NO_CACHE"},
    "logsConfig": {
        "cloudWatchLogs": {
            "status": "ENABLED",
            "groupName": log_group
        }
    }
}

files = {
    "trust-policy.json": trust,
    "service-policy.json": policy,
    "project.json": config
}

for name, content in files.items():
    path = out / name
    path.write_text(json.dumps(content, indent=2) + "\n")
    print("Prepared:", path)
PY

echo
echo "=== CONFIGURATION ==="
echo "Project: $PROJECT"
echo "IAM role: $ROLE"
echo "S3 bucket: $BUCKET"
echo "S3 object: $KEY"
echo "Build environment: standard:8.0 / MEDIUM"
echo "ECR publishing: disabled"

if [[ "$MODE" == "--check" ]]; then
  echo
  echo "CHECK PASSED: No AWS resources were changed."
  exit 0
fi

echo
echo "=== 1. Create CodeBuild service role ==="

if aws iam get-role \
  --role-name "$ROLE" >/dev/null 2>&1; then
  echo "Role already exists: $ROLE"
else
  aws iam create-role \
    --role-name "$ROLE" \
    --assume-role-policy-document \
      "file://$OUT/trust-policy.json" \
    --description "Service role for Qwen3 V1 CodeBuild" \
    --output json >/dev/null
fi

aws iam put-role-policy \
  --role-name "$ROLE" \
  --policy-name "qwen3-v1-codebuild-permissions" \
  --policy-document "file://$OUT/service-policy.json"

echo "Service role configured."

echo
echo "=== 2. Create private source bucket ==="

EXISTING=$(aws s3api list-buckets \
  --query "Buckets[?Name=='${BUCKET}'].Name" \
  --output text)

if [[ "$EXISTING" == "$BUCKET" ]]; then
  echo "Source bucket already exists."
else
  aws s3api create-bucket \
    --bucket "$BUCKET" \
    --region "$REGION" \
    --create-bucket-configuration \
      "LocationConstraint=$REGION"
fi

aws s3api put-public-access-block \
  --bucket "$BUCKET" \
  --region "$REGION" \
  --public-access-block-configuration \
    'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true'

aws s3api put-bucket-encryption \
  --bucket "$BUCKET" \
  --region "$REGION" \
  --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

echo
echo "=== 3. Upload source archive ==="

aws s3 cp "$SOURCE" \
  "s3://$BUCKET/$KEY" \
  --region "$REGION"

aws s3api head-object \
  --bucket "$BUCKET" \
  --key "$KEY" \
  --region "$REGION" \
  --query '{Size:ContentLength,Encryption:ServerSideEncryption}' \
  --output json

echo
echo "=== 4. Create CodeBuild project ==="

EXISTING_PROJECT=$(aws codebuild batch-get-projects \
  --names "$PROJECT" \
  --region "$REGION" \
  --query 'projects[0].name' \
  --output text)

if [[ "$EXISTING_PROJECT" == "$PROJECT" ]]; then
  echo "Project already exists. Not overwriting configuration."
else
  aws codebuild create-project \
    --region "$REGION" \
    --cli-input-json "file://$OUT/project.json" \
    --query 'project.{Name:name,Arn:arn}' \
    --output json
fi

echo
echo "=== SETUP COMPLETE ==="
echo "CodeBuild project: $PROJECT"
echo "No build was started."
echo "No Kubernetes resources were modified."
