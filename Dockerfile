# Use the base image of python 3.12 from Docker Hub
FROM python:3.12

# Set environment variables
# Prevents python from writing '.pyc' files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure that the python output is sent directly to the terminal without being buffered
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container /workspace
WORKDIR /workspace

#
RUN apt-get update && apt-get install -y

#
RUN pip install --upgrade pip

#
COPY requirements.txt .

#
RUN pip install -r requirements.txt

#
CMD ["sleep", "infinity"]
