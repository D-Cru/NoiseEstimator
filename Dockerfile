FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple noiseestimator==0.0.5

# Install the Streamlit library
RUN pip install streamlit


# Set the working directory in the container
WORKDIR /app

COPY app.py /app/app.py
COPY README.md /app/README.md
COPY data /app/data

# The EXPOSE instruction informs Docker that the container listens on the specified network ports at runtime. Your container needs to listen to Streamlit’s (default) port 8501:
EXPOSE 8501

# The HEALTHCHECK instruction tells Docker how to test a container to check that it is still working. Your container needs to listen to Streamlit’s (default) port 8501:
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# An ENTRYPOINT allows you to configure a container that will run as an executable. Here, it also contains the entire streamlit run command for your app, so you don’t have to call it from the command line:
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker

# Based on your server's network configuration, you could map to port 80/443 so that users can view your app using the server IP or hostname. For example: http://your-server-ip:80 or http://your-hostname:443.