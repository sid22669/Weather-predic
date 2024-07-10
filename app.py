#importing packages
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    precipitation = float(request.form['precipitation'])
    temp_max = float(request.form['temp_max'])
    temp_min = float(request.form['temp_min'])
    wind = float(request.form['wind'])
    
    # Make a prediction using the loaded model
    prediction = model.predict([[precipitation, temp_max, temp_min, wind]])
    
    # Pass the prediction result to the result.html page
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
