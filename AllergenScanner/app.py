from flask import Flask, request, jsonify, render_template, url_for, session

app = Flask(__name__)

DEBUG = True

# ---------------------- SCRIPT -----------------------
import base64
import requests

import time

# OpenAI API Key
try:
    with open("openai_key", "r") as key_file:
        api_key = key_file.read().strip()
except FileNotFoundError:
    print("No API key found.")
    
allergens = {
    0: "gluten",
    1: "eggs",
    2: "milk",
    3: "nuts",
    4: "peanuts",
    5: "soja",
    6: "molluscs",
    7: "fish",
    8: "lupin",
    9: "crustaceans",
    10: "sesame",
    11: "mustard",
    12: "celery",
    13: "sulphites"
}

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to format the answer given by the model
def format_answer(input_string):
    components = input_string.split()

    # Initialize an empty dictionary to store the allergy information
    allergies = {}

    # Iterate through the components, assuming each pair is of the form "Allergy: Yes/No"
    for i in range(0, len(components), 2):
        allergy = components[i].rstrip(":")
        answer = components[i + 1]

        # Map "Yes" to True and "No" to False
        allergies[allergy] = answer.lower() == "yes"

    # Create the formatted string
    formatted_string = ", ".join(f"{allergy}: {'Yes' if value else 'No'}" for allergy, value in allergies.items())

    return formatted_string

def predict_all_allergens(image_data):
    prediction = {}  # Initialize an empty dictionary

    if DEBUG is True:
        time.sleep(2)

        prediction["prediction"] = "Gluten: Yes, Eggs: Yes, Milk: Yes, Nuts: Yes, Peanuts: No, Soja: Yes, Molluscs: No, Fish: Yes, Lupin: No, Crustaceans: Yes, Sesame: No, Mustard: No, Celery: Yes, Sulphites: Yes"

        return prediction
    else:

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "text",
                        "text": f"Does this product contain gluten, eggs, milk, nuts, peanuts, soja, molluscs, fish, lupin, crustaceans, sesame, mustard, celery or sulphites? Answer only by 'Yes' or 'No' for each allergen in the following way: Gluten: Yes/No, Eggs: Yes/No, Milk: Yes/No, Nuts: Yes/No, Peanuts: Yes/No, Soja: Yes/No, Molluscs: Yes/No, Fish: Yes/No, Lupin: Yes/No, Crustaceans: Yes/No, Sesame: Yes/No, Mustard: Yes/No, Celery: Yes/No, Sulphites: Yes/No."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image_data}",
                            "detail": "auto"
                        }
                    }
                    ]
                }
            ],
            "max_tokens": 1000
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) 
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']

        content = format_answer(content)

        prediction["prediction"] = content

        return prediction
# ---------------------- SCRIPT END -----------------------

@app.route('/api/process-image', methods=['POST'])
def predict_allergens_endpoint():
    try:
        # Get the image data from the request (assuming it's sent as raw bytes)

        data = request.json  # This will automatically parse the JSON data
        image_data = data['image']  # Make sure this matches the key you used in your JavaScript fetch request

        prediction = predict_all_allergens(image_data)

        # Return the prediction as JSON
        return jsonify(prediction)

    except Exception as e:
        # Handle any exceptions (e.g., invalid input, API failures)
        return jsonify({'error': str(e)})
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/prediction.html')
def prediction_results():
    # Retrieve the prediction from the session
    prediction = session.get('prediction', None)

    # Pass the prediction data to the prediction.html template
    return render_template("prediction.html", prediction=prediction)

if __name__ == '__main__':
     app.run(debug=True)  # Run the app in debug mode