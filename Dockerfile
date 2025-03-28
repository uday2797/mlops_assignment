FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY starter.py .

COPY model2.bin .

RUN pip install pandas pyarrow scikit-learn

ENTRYPOINT ["python", "starter.py"]