import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from flask import Flask, request, render_template
app = Flask(__name__)

# Load and preprocess the data
data = pd.read_csv("bridge_data.csv")

# Map categorical variables to numeric values
data["Material"] = data["Material"].map({"Steel": 0, "Concrete": 1, "Wood": 2})
data["Weather_Conditions"] = data["Weather_Conditions"].map({"Sunny": 0, "Windy": 1, "Rainy": 2, "Cloudy": 3, "Snowy": 4})
data["Construction_Quality"] = data["Construction_Quality"].map({"Bad": 0, "Good": 1})
data["Bridge_Design"] = data["Bridge_Design"].map({"Arch": 0, "Beam": 1, "Truss": 2})
data["Collapse_Status"] = data["Collapse_Status"].map({"Standing": 0, "Collapsed": 1})

# Select features (X) and target (y)
# Define the feature order
feature_order = ['Material', 'Weather_Conditions', 'Construction_Quality', 'Bridge_Design', 
                 'Length (m)', 'Age (years)', 'Traffic_Volume (vehicles/day)', 
                 'Width (m)', 'Height (m)', 'Water_Flow_Rate (m³/s)', 
                 'Stress (MPa)', 'Strain (%)', 'Tensile_Strength (MPa)', 
                 'Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)']

x = data[feature_order]  # Ensure x is ordered according to feature_order
y = data["Collapse_Status"]

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
def preprocess_bridge_data(data):
    # Transform incoming data to match the format and scale expected by the model
    # Example: apply scaling, encoding, etc.
    features = [data['Material'], data['Weather_Conditions'], data['Construction_Quality'], data['Bridge_Design'],
                data['Length'], data['Age'], data['Traffic_Volume'], data['Width'], data['Height'],
                data['Water_Flow_Rate'], data['Stress'], data['Strain'], data['Tensile_Strength'],
                data['Rainfall'], data['Temperature'], data['Humidity']]
    
    # Apply scaler transformation if used during model training
    features = scaler.transform([features])
    return features
# Check if model exists, then load it or create and train a new one
if os.path.exists("model.h5"):
    model = load_model("model.h5")
else:
    # Create a new model
    model = Sequential()
    model.add(Dense(32, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=0)

    # Save the model
    model.save("model.h5")

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/map')
def map():
    return render_template('map.html')
from flask import jsonify

# Route to get bridge data with predictions
@app.route('/get_bridges')
def get_bridges():
    try:
        # Load bridge data
        bridge_data = pd.read_csv("bridges_coordinates.csv")  # Replace with your actual CSV
        
        # Filter out collapsed or unidentified bridges
        filtered_bridges = bridge_data[
            (bridge_data['status'] != 'Collapsed')
        ]
        
        # Prepare JSON response
        results = []
        for _, row in filtered_bridges.iterrows():
            results.append({
                "lat": row["latitude"],
                "lng": row["longitude"],
                "status": row["status"]
            })
        
        return jsonify(results)
    
    except Exception as e:
        print("Error loading bridge data:", e)
        return jsonify({"error": "Failed to load bridge data"}), 500


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data and convert them to the correct types
        material = int(request.form['Material'])
        weather_conditions = int(request.form['Weather_Conditions'])
        construction_quality = int(request.form['Construction_Quality'])
        bridge_design = int(request.form['Bridge_Design'])
        length = float(request.form['Length (m)'])  # Ensure correct data types
        age = float(request.form['Age (years)'])     # Ensure correct data types
        traffic_volume = float(request.form['Traffic_Volume (vehicles/day)'])  # Ensure correct data types
        width = float(request.form['Width (m)'])
        height = float(request.form['Height (m)'])
        water_flow_rate = float(request.form['Water_Flow_Rate (m³/s)'])
        stress = float(request.form['Stress (MPa)'])
        strain = float(request.form['Strain (%)'])
        tensile_strength = float(request.form['Tensile_Strength (MPa)'])
        rainfall = float(request.form['Rainfall (mm)'])
        temperature = float(request.form['Temperature (°C)'])
        humidity = float(request.form['Humidity (%)'])

        # Prepare input data as a DataFrame in the correct order
        input_data = pd.DataFrame([[material, weather_conditions, construction_quality, bridge_design,
                                     length, age, traffic_volume, width, height,
                                     water_flow_rate, stress, strain, tensile_strength,
                                     rainfall, temperature, humidity]], 
                                   columns=feature_order)  # Ensure correct column order
        
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data_scaled)
        collapse_probability = prediction[0][0]
        status = "Collapsed" if collapse_probability > 0.5 else "Standing"

        # Render prediction result
        return render_template('index.html', 
                               prediction_text=f"Probability of collapse: {collapse_probability:.2f}. Predicted Status: {status}")

    except Exception as e:
        return str(e)
        

if __name__ == "__main__":
    app.run(debug=True,port=8500)
