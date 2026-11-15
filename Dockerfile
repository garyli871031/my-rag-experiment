# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for building certain Python packages if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for ultra-fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency specification first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies using uv (falls back to standard pip behavior via requirements.txt)
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Run your Streamlit app (adjust app.py / v2app.py to match your main script)
CMD ["streamlit", "run", "v2app.py"]