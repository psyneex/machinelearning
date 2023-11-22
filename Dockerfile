# Use an official Python runtime as a parent image
FROM python:3.11


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install any needed packages specified in requirements.txt
RUN pip install -r /app/requirements.txt

RUN pip install uvicorn

# Make port 80 available to the world outside this container
EXPOSE 8000

# Run the app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
