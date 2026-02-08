# Use the official Python base image
FROM python:3.7-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any dependencies
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the Python script into the container
COPY my_function.py .

# Run the Python script when the container is started
CMD ["python", "app.py"]