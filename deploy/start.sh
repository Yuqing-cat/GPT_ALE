# Start nginx
nginx

# start sshd
/usr/sbin/sshd

# set env
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
export MLFLOW_S3_ENDPOINT_URL="http://0.0.0.0:9000"

# Start mlflow
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 -p 5000 --default-artifact-root s3://mlflow

# Start uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 10

