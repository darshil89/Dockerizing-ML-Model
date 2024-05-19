from fastapi import FastAPI, Request
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import uvicorn

app = FastAPI()

# Load tokenizer and model
model = AutoModelForSequenceClassification.from_pretrained("climatebert/distilroberta-base-climate-sentiment")
tokenizer = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-sentiment")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the API!"}

@app.post("/predict")
async def predict(request: Request):
    
    data = await request.json()
    if "text" in data:
        user_input = data["text"]
        
        # Tokenize input
        inputs = tokenizer([user_input], padding="max_length", truncation=True, return_tensors="pt", max_length=512)
        
        # Ensure the model is in evaluation mode
        model.eval()
        
        # Perform prediction within torch.no_grad() context to save memory and computation
        with torch.no_grad():
            output = model(**inputs)
            y_pred = np.argmax(output.logits.numpy(), axis=1).tolist()  # Convert to list for JSON serialization

        response = {"Received Text": user_input, "Prediction": y_pred}
    else:
        response = {"Received Text": "No text received."}
    
    return response

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8080, reload=True)
