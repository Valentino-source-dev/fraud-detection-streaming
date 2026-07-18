FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.15.1 psycopg2-binary boto3

EXPOSE 5000

CMD ["sh", "-c", \
     "mlflow server \
        --backend-store-uri ${BACKEND_STORE_URI} \
        --default-artifact-root ${ARTIFACT_ROOT} \
        --artifacts-destination /mlflow-artifacts \
        --host 0.0.0.0 \
        --port 5000 \
        --serve-artifacts"]
