# Email Ticket Classification using FastAPI and DistilBERT

## Project Overview

This project aims to classify customer support tickets into predefined categories based on the text content of the ticket. The project utilizes a **DistilBERT** model to classify tickets and is served via a **FastAPI** application. The model predicts the appropriate category (`queue`) for each ticket by analyzing the `subject` and `body` of the ticket.

## Features

- **Multilingual Text Classification**: Utilizes DistilBERT to handle customer support tickets in multiple languages.
- **FastAPI Integration**: The model is served via a RESTful API built with FastAPI.
- **Dockerized Application**: The project is containerized using Docker for easy deployment.
- **Continuous Integration (CI)**: Includes automated testing and deployment using GitHub Actions.

## Technologies Used

- **Python 3.11**
- **Transformers (Hugging Face)**
- **DistilBERT (Multilingual)**
- **FastAPI**
- **PyTorch**
- **Docker**
- **GitHub Actions (CI/CD)**

## Project Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Email-ticket-classification-.git
cd Email-ticket-classification-
````
### 2. Install Dependencies
```bash
pip install -r requirements.txt
````
### 3. Run the Application
```bash
uvicorn app:app --reload
````
The application will be accessible at http://127.0.0.1:8000/.
### 4. Run the Tests
```bash
pytest tests/test_app.py
````
## Model Training
The DistilBERT model was fine-tuned using the helpdesk_customer_tickets.csv dataset. The training process can be run using the following command:
```bash
python train_model.py --data_path './helpdesk_customer_tickets.csv' --output_dir './models' --epochs 10 --batch_size 16
````
The trained model is saved in the ./models directory and can be loaded for inference.

## API Endpoints
POST /predict: Predicts the category of a customer support ticket.
Request Body:
  {
  "text": "Your ticket content here"
}
Response:
{
  "text": "Your ticket content here",
  "predicted_class_name": "Product Support"
}

## Docker Setup
### 1. Build Docker Image
```bash
docker build -t email-ticket-classification .
````
### 2. Run Docker Container
```bash
docker run -p 8000:8000 email-ticket-classification
````
The application will be available at http://127.0.0.1:8000/.

## Continuous Integration (CI)
This project includes a CI pipeline using GitHub Actions. The pipeline automatically runs the test suite whenever changes are pushed to the repository. It also builds the Docker image when a new tag is created.

Future Improvements
Model Optimization: Further fine-tuning of the model could improve performance on certain categories.
Deployment: Deploy the model on a cloud platform (e.g., AWS, Heroku) for production use.
API Enhancements: Add more endpoints for model management (e.g., loading different models).













