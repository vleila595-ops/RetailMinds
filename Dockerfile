FROM python:3.10.6

WORKDIR /app

COPY . .

# upgrade pip
RUN pip install --upgrade pip

# install system dependencies (IMPORTANT for prophet & pmdarima)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN pip install -r requirements.txt

# run your app
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:10000"]
