FROM continuumio/miniconda3:latest

# RUN pip install mlflow boto3 pymysql
RUN pip install -r requirements.txt
RUN python -m spacy download pt_core_news_sm

ADD . /app
WORKDIR /app

COPY wait-for-it.sh wait-for-it.sh 
RUN chmod +x wait-for-it.sh