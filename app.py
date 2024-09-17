from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import json
from fastapi.responses import JSONResponse

app = FastAPI()

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('./models')
model = DistilBertForSequenceClassification.from_pretrained('./models')
model.eval()

# Load label mappings
with open('./models/id2label.json', 'r') as f:
    id2label = json.load(f)

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the request body model
class TextInput(BaseModel):
    text: str

# Preprocessing function to tokenize text
def preprocess(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    return inputs

# Prediction endpoint
@app.post('/predict')
async def predict(input: TextInput):
    try:
        text = input.text
        inputs = preprocess(text)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = torch.argmax(logits, dim=-1).item()

        predicted_class = id2label[str(predicted_class_idx)]

        return {
            "text": text,
            "predicted_class_name": predicted_class
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
