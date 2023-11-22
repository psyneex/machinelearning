import numpy as np
import pickle


# Load the model from a file
model = pickle.load(open("model1.pkl", "rb"))

# Define a prediction function that takes a list of features as input and returns a list of predictions as output
def predict(features):
    # Convert the features to a numpy array
    features = np.array(features)
    # Make predictions using the model
    predictions = model.predict(features)
    # Convert the predictions to a list
    predictions = predictions.tolist()
    # Return the predictions
    return predictions