from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
app = Flask(__name__)

# Load the model
model = joblib.load("model.pkl")


# Render the form page
@app.route("/")
def form():
    return render_template("predict.html")


# Handle form submission and provide prediction
@app.route("/predict", methods=["POST"])
def predict():
    # print("Hello World")
    # Get form data
    features = []
    try:
        # Age
        features.append(int(request.form["age"]))

        # Gender
        gender = request.form["gender"]
        features.append(1 if gender == "Female" else 0)

        # Marital Status
        mar = request.form["marital_status"]
        features.append(0 if mar == "Married" else 1 if mar == "Single" else 2)

        # Occupation
        occ = request.form["occupation"]
        features.append(1 if occ in ["Employee", "Self Employeed"] else 0)

        # Monthly Income
        inc = request.form["monthly_income"]
        features.append(0 if inc == "No Income" else 1)

        # Educational Qualifications
        edu = request.form["educational_qualifications"]
        ans = 0
        if edu == "Graduate":
            ans = 1
        elif edu == "Post Graduate":
            ans = 2
        elif edu == "Ph.D":
            ans = 3
        elif edu == "School":
            ans = 4
        else:
            ans = 5

        features.append(ans)

        # Family Size
        features.append((request.form["family_size"]))

        # Latitude
        features.append(float(request.form["latitude"]))

        # Longitude
        features.append(float(request.form["longitude"]))

        # Pin Code
        features.append((request.form["pin_code"]))

        # Output
        out = request.form["output"]
        features.append(0 if out == "No" else 1)

    except:
        pass
    # Make prediction
    from sklearn.preprocessing import StandardScaler
    import joblib

    # scaler = joblib.load('scaler.pkl')

    # X_test_scaled = scaler.transform([features])
    features= np.array(features)
    prediction = model.predict(features.reshape(1,-1))
    # print(prediction)
    
    if(prediction == [1]):
        ans = "Positive"
    else:
        ans="Negative"
    # Render the result page with prediction
    return render_template("result.html", prediction=ans)


print("Hello")

if __name__ == "__main__":
    app.run(debug=True)
