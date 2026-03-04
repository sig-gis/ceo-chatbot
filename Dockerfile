FROM python:3.10-trixie

WORKDIR /app

# INSTALL GCLOUD SDK ##############
# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH="$PATH:/usr/local/gcloud/google-cloud-sdk/bin"

# Copy the requirements file into the container at /app
COPY . /app/

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size, and --trusted-host is good practice
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy the application source code into the container
RUN pip install .

ENTRYPOINT [ "streamlit", "run", "demo/chat_app.py", "--server.port", "8080" ]
