from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# 1. Initialize the API app
app = FastAPI(title="HAS AI Routing API")

# 2. Load your trained model (the ".pkl" file)
print("Loading AI Model into memory...")
model = joblib.load('hostel_routing_model.pkl')

# 3. Define the structure of the incoming data
class ComplaintRequest(BaseModel):
    text: str

# 4. Create the main prediction endpoint
@app.post("/predict")
async def predict_complaint(request: ComplaintRequest):
    complaint_text = request.text
    
    # Ask the model to predict the role and its confidence level
    predicted_role = model.predict([complaint_text])[0]
    confidence = model.predict_proba([complaint_text]).max()
    
    # Return the JSON response
    return {
        "text": complaint_text,
        "assigned_role": predicted_role,
        "confidence": float(confidence)
    }

# 5. A simple health-check endpoint
@app.get("/")
async def root():
    return {"status": "Online", "message": "HAS AI Router is running!"}