#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "python-dotenv"]
# ///
"""Probe write permission on source.coop with current .env credentials."""
import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    region_name=os.environ.get("SOURCE_COOP_REGION", "us-west-2"),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    aws_session_token=os.environ.get("AWS_SESSION_TOKEN") or None,
    config=Config(signature_version="s3v4"),
)

bucket = os.environ["SOURCE_COOP_BUCKET"]
prefix = os.environ["SOURCE_COOP_PREFIX"].rstrip("/") + "/"

print(f"Access key:  {os.environ['AWS_ACCESS_KEY_ID']}")
print(f"Target:      s3://{bucket}/{prefix}")

# 1. List
try:
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=3)
    print(f"\nlist_objects_v2 OK — {resp.get('KeyCount', 0)} keys at {prefix}")
except ClientError as e:
    print(f"\nlist_objects_v2 FAIL: {e.response['Error'].get('Code')}")

# 2. PUT
key = f"{prefix}_probe_{os.getpid()}.txt"
try:
    s3.put_object(Bucket=bucket, Key=key, Body=b"probe")
    print(f"put_object OK -> {key}")
    s3.delete_object(Bucket=bucket, Key=key)
    print("delete_object OK -- credentials have write access")
except ClientError as e:
    print(f"put_object FAIL: {e.response['Error'].get('Code')} - {e.response['Error'].get('Message')}")
    print("\n>>> credentials are READ-ONLY for this prefix.")
    print(">>> Generate WRITE-scoped keys in the source.coop UI:")
    print(">>>   https://source.coop/repositories/e4drr-project/forecasts/manage")
