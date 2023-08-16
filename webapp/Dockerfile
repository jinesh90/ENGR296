# Use the official Python image
FROM python:3.10

# Set the working directory
WORKDIR .

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the app files into the container
COPY . .

# Expose port 5000 for Flask
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
