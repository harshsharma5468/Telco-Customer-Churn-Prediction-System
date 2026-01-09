# Use a lightweight Python 3.12 image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for XGBoost, Jupyter, and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

RUN apt-get update && apt-get install -y dos2unix && rm -rf /var/lib/apt/lists/*

RUN chmod +x run_pipeline.sh run_training.sh run_inference.sh



# Expose Streamlit's default port
EXPOSE 8501

# Execute the full pipeline on startup
ENTRYPOINT ["/bin/bash", "./run_pipeline.sh"]