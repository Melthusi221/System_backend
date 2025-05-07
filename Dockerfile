# Use official Python image as base
FROM python:3.9

# Set working directory inside container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the application port (Render automatically assigns the PORT)
EXPOSE 10000

# Define environment variables
ENV PORT=10000

# Ensure persistent storage directory exists
RUN mkdir -p /data/models

# Run the Flask application
CMD ["python", "app.py"]
