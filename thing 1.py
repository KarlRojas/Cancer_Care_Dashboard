import numpy as np
import json
import requests  # Add the import statement for the 'requests' library
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import shap

BASE_URL = 'https://test/fhir'

def patient_data(patient_id) :
    try:
        response = requests.get('{}/{}/{}'.format(BASE_URL, 'Patient', patient_id))
        if response.status_code == 200:
            # Normalize and select relevant columns
            patient_df = pd.json_normalize(response.json())[['id', 'gender', 'birthDate']]
            # Convert birthDate column to datetime
            patient_df = patient_df.astype({'birthDate': 'datetime64[ns]'})
            print(patient_df.info())
            return patient_df
        else:
            print("Failed to fetch patient data. Status code:", response.status_code)
            return None
    except requests.expections.RequestsException as e:
        print("Error", e)
        return None


patient_id = 2
patient_test = patient_data(patient_id)
print(patient_test)

