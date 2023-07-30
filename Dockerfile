FROM python:3.10.6

WORKDIR /app
USER josa

RUN pip install --upgrade pip

ADD requirements.txt /app/requirements.
RUN pip install -r requirements.txt


ADD . /app

ENTRYPOINT [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]