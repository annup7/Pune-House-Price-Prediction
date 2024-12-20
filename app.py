from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np
import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Load model and columns
with open('pune_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']
    locations = data_columns[4:]  # Extract location names for the dropdown

def connect_db():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )

def predict(sqft, bhk, balcony, bath, location):
    loc_index = -1
    if location in data_columns:
        loc_index = data_columns.index(location.lower())  # Find the index of the location in data columns

    # Prepare the input array with zeros
    X = np.zeros(len(data_columns))
    X[0] = sqft
    X[1] = bath
    X[2] = balcony
    X[3] = bhk
    if loc_index >= 0:
        X[loc_index] = 1  # Set the location feature if found

    # Return the predicted price
    return round(model.predict([X])[0], 2)

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_price = 0
    if request.method == 'POST':
        try:
            sqft = float(request.form['sqft'])
            bhk = int(request.form['bhk'])
            balcony = int(request.form['balcony'])
            bath = int(request.form['bath'])
            location = request.form['location']

            # Predict the price
            predicted_price = predict(sqft, bhk, balcony, bath, location)

            # Store the prediction in the MySQL database
            db = connect_db()
            cursor = db.cursor()
            cursor.execute(
                "INSERT INTO predictions (sqft, bhk, balcony, bath, location, predicted_price) VALUES (%s, %s, %s, %s, %s, %s)",
                (float(sqft), int(bhk), int(balcony), int(bath), location, float(predicted_price)))
            db.commit()
            cursor.close()
            db.close()

            # Return the predicted price as JSON
            return jsonify({'success': True, 'price': predicted_price})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

    return render_template('index.html', locations=locations, predicted_price=predicted_price)

if __name__ == '__main__':
    app.run()
