import pandas as pd 
import numpy as np 
from flask import Flask, render_template, request
import pickle


app = Flask(__name__)
data = pd.read_csv('house_cleaned_data.csv')  # Ensure this file exists and is accessible
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))  # Ensure this model file exists

@app.route('/')
def index():
    locations = sorted(data['location'].unique())  # Get unique sorted locations from the dataset
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get('location')
        bhk = float(request.form.get('bhk'))
        bath = float(request.form.get('bath'))
        sqft = float(request.form.get('total_sqft'))  # Ensure sqft is a float

        # Check if any values are empty or invalid
        if not location or bhk <= 0 or bath <= 0 or sqft <= 0:
            return "Invalid input values, please check again."

        # Prepare the input data for prediction
        input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        
        # Get the price prediction from the model
        prediction = pipe.predict(input)[0] * 1e5  # Multiply by 100,000 as per your model scaling

        return str(np.round(prediction, 2))  # Return the prediction rounded to 2 decimal places

    except Exception as e:
        return str(e)  # Return any errors encountered as a string for debugging

if __name__ == '__main__':
    app.run(debug=True)
