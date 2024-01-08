FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10-slim
WORKDIR /app

COPY ./env_requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./nltk_data /app/nltk_data
COPY ./gard_id_names_synonyms_2024.json /app
COPY ./LICENSE /app
COPY ./*.py /app
