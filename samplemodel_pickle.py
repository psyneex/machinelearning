import numpy as np
import pickle
from samplemodel_train import sample_model


with open("model1.pkl","wb") as f:
    model = pickle.dump("sample_model",f)




