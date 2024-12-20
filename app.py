from flask import Flask, request, render_template, jsonify
import pickle
import json
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load the trained model and columns
with open('pune_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']
    locations = data_columns[4:]  # Extract location names for dropdown

# Ensure the database table `predictions` has the following schema:
# CREATE TABLE predictions (
#     id SERIAL PRIMARY KEY,
#     sqft FLOAT,
#     bhk INT,
#     balcony INT,
#     bath INT,
#     location TEXT,
#     predicted_price FLOAT,
#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );

def connect_db():
    """Connect to the PostgreSQL database."""
    try:
        connection = psycopg2.connect(dsn=os.getenv('DATABASE_URL'), cursor_factory=RealDictCursor)
        print("Database connected successfully!")
        return connection
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def insert_prediction(db, total_sqft, bhk, balcony, bath, location, predicted_price):
    """Insert prediction into the database."""
    try:
        cursor = db.cursor()
        query = """
            INSERT INTO predictions (sqft, bhk, balcony, bath, location, predicted_price)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (total_sqft, bhk, balcony, bath, location, predicted_price))
        db.commit()
        cursor.close()
        print("Prediction inserted successfully!")
    except Exception as e:
        print(f"Error inserting into the database: {e}")
        db.rollback()

def predict_price(total_sqft, bhk, balcony, bath, location):
    """Predict house price based on input features."""
    try:
        # Initialize input array with zeros
        X = np.zeros(len(data_columns))
        X[0] = total_sqft
        X[1] = bath
        X[2] = balcony
        X[3] = bhk

        # Handle location encoding
        if location in locations:
            loc_index = data_columns.index(location)
            X[loc_index] = 1

        # Convert to DataFrame with correct column names
        X_df = pd.DataFrame([X], columns=data_columns)

        # Predict price
        return round(model.predict(X_df)[0], 2)
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_price = None
    error_message = None

    if request.method == 'POST':
        try:
            # Retrieve form inputs and validate them
            total_sqft = float(request.form.get('total_sqft', 0))
            bhk = int(request.form.get('bhk', 0))
            balcony = int(request.form.get('balcony', 0))
            bath = int(request.form.get('bath', 0))
            location = request.form.get('location', '').strip().title()  # Normalize location

            # Validate inputs
            if total_sqft <= 0 or bhk <= 0 or bath <= 0:
                raise ValueError("Square footage, BHK, and bathrooms must be greater than zero.")
            if location not in locations:
                raise ValueError(f"Location '{location}' is not available. Please select a valid location.")

            # Predict house price
            predicted_price = predict_price(total_sqft, bhk, balcony, bath, location)

            # Store prediction in the database
            db = connect_db()
            if db:
                insert_prediction(db, total_sqft, bhk, balcony, bath, location, predicted_price)
                db.close()

            # Return the predicted price as JSON
            return jsonify({'success': True, 'price': predicted_price})

        except ValueError as ve:
            error_message = str(ve)
            print(f"Validation error: {ve}")
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            print(f"Error: {e}")
            return jsonify({'success': False, 'message': error_message})

    return render_template('index.html', locations=locations, predicted_price=predicted_price, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
