import pandas as pd 
import numpy as np 
from flask import Flask, render_template, request
import pickle


app = Flask(__name__)
data = pd.read_csv('house_cleaned_data.csv') 
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))  

@app.route('/')
def index():
    locations = sorted(data['location'].unique()) 
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get('location')
        bhk = float(request.form.get('bhk'))
        bath = float(request.form.get('bath'))
        sqft = float(request.form.get('total_sqft'))  

        if not location or bhk <= 0 or bath <= 0 or sqft <= 0:
            return "Invalid input values, please check again."

        input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        

        prediction = pipe.predict(input)[0] * 1e5 

        return str(np.round(prediction, 2))  

    except Exception as e:
        return str(e) 

if __name__ == '__main__':
    app.run(debug=True)
