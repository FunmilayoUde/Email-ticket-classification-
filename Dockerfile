# --- Stage 1: Training the Model ---
    FROM python:3.9-slim AS train

    # Set working directory
    WORKDIR /app
    
    # Copy necessary files for training
    COPY requirements.txt ./
    COPY ttrain.py ./
    COPY helpdesk_customer_tickets.csv ./
    

    
    # Install dependencies
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Run the training script
    RUN python ttrain.py --data_path "./helpdesk_customer_tickets.csv" --output_dir "./models" --epochs 10 --batch_size 8
    
    # --- Stage 2: Inference ---
    FROM python:3.9-slim AS inference
    
    # Set working directory
    WORKDIR /app
    
    # Copy only the necessary files from the previous stage
    # Copy the trained model and tokenizer from the training stage
    COPY --from=train /app/models ./models
    COPY requirements.txt ./
    COPY app.py ./
    
    # Install only the necessary dependencies for inference
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Expose port for FastAPI
    EXPOSE 8000
    
    # Command to run the FastAPI app
    CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
    