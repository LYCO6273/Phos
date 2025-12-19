#!/bin/bash
# Install dependencies if needed (assuming they are installed in the environment)
# pip install -r requirements.txt

# Run the FastAPI app using uvicorn on port 8086
# Reload is enabled for development
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8086 --reload
