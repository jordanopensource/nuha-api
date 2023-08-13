FROM python:3.10.6-slim AS builder

WORKDIR /app

RUN pip install --upgrade pip
ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
COPY . /app


# Run stage
FROM python:3.10.6-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages  /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /app .

ENTRYPOINT [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]

