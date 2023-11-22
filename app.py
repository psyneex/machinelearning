from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from samplemodel_predict import predict

app = FastAPI()

class features(BaseModel):
    features: List[list[float]]
    
class predictions(BaseModel):
    predictions: list[float]
    
# Define the endpoint for the prediction function.
@app.post("/predict", response_model=predictions)
def predict_endpoint(features: features):
    # Get the features from the request body
    features = features.features
    # Validate the features
    if not features or not all(len(feature) == len(features[0]) for feature in features):
        # Raise an exception if the features are empty or have inconsistent lengths
        raise HTTPException(status_code=400, detail="Invalid features")
    # Call the prediction function
    predictions = predict(features)
    # Return the predictions as a response
    return predictions(predictions=predictions)