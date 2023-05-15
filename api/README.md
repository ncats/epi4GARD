# Set up conda environment

`conda create --name epiapi-042823 python=3.10.11`

`conda activate epiapi-042823`

`conda install transformers nltk pandas gunicorn uvicorn streamlit more-itertools unidecode`

`conda install -c conda-forge fastapi`

# Activate python environment
<!---source /opt/miniconda3/bin/activate--->

`conda activate epiapi-042823`

# To Start the gunicorn server in development mode:
`gunicorn app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:9000 --reload &2>log`

OR if that doesn't work then this: 

`gunicorn app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 127.0.0.1:9000 --reload &2>log`

OR if that doesn't work then this: 

`gunicorn app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 127.0.0.1:9000 --keyfile uvicorn.key --certfile uvicorn.cert --reload &2>log`


# Previous instructions to start the gunicorn server in production mode (Ignore below this):
`gunicorn app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 10.9.1.108 --keyfile uvicorn.key --certfile uvicorn.cert &2>log`

#to start the uvicorn server:
`nohup uvicorn --host 10.9.1.108 --ssl-keyfile=./uvicorn.key --ssl-certfile=./uvicorn.cert mytest:app --reload & 2>log`

OR

`nohup uvicorn --host 10.9.1.108 --ssl-keyfile=./uvicorn.key --ssl-certfile=./uvicorn.cert app:app --reload & 2>log`

`source /workspace/anaconda3/bin/activate`

`conda activate /workspace/condaEnv/fastapi`

`nohup uvicorn --host 10.9.1.108 --port 8889 --ssl-keyfile=./uvicorn.key --ssl-certfile=./uvicorn.cert app:app --reload & 2>log`

`gunicorn app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 10.9.1.108 --keyfile uvicorn.key --certfile uvicorn.cert --reload &2>log`

Last commands that worked:
`nohup uvicorn --host 127.0.0.1 --port 9000 api-app-v4:head_app --reload & 2>log`
