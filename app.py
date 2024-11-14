import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model, with error handling if the model file is missing
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: The model file 'model.pkl' was not found.")
    model = None

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from form and convert to floats
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]

        # Define feature names for the DataFrame
        features_name = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                         'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                         'Oldpeak', 'ST_Slope']
        
        # Create a DataFrame for prediction
        df = pd.DataFrame(features_value, columns=features_name)
        
        # Ensure model is loaded before prediction
        if model:
            output = model.predict(df)[0]  # Assume the model returns an array-like structure
            
            # Interpret prediction result
            if output == 1:
                res_val = "The Patient Has Heart Disease, please consult a Doctor."
            else:
                res_val = "The Patient is Normal."
        else:
            res_val = "Model is not loaded properly. Prediction cannot be made."
        
    except Exception as e:
        # In case of an error during prediction, capture and display it
        res_val = f"Error in prediction: {str(e)}"
    
    return render_template('index1.html', prediction_text='Result - {}'.format(res_val))

if __name__ == "__main__":
    app.run(debug=True)  # Enabling debug mode for detailed error logs
