FROM python:3.10-slim 

WORKDIR /app 
COPY requirements.txt visualizer.py util.py fake_engineered_features.csv xgboost_model.pkl ./
RUN pip install -r requirements.txt 

CMD ["python", "visualizer.py"]
