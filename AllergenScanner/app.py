from flask import Flask, request, jsonify, render_template, url_for

app = Flask(__name__)

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

# Allergen number from 0 to 13
def predict_allergen(image_data, allergen):

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
                "text": f"Does this product contain {allergens[allergen]}? Answer only by 'Yes' or 'No'"
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
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) 
    response_json = response.json()
    content = response_json['choices'][0]['message']['content']
    return content

def predict_all_allergens(image_data):
    prediction = {}  # Initialize an empty dictionary

    # prediction[allergens[0]] = predict_allergen(image_data, 0)
    for allergen in range(len(allergens)):
        prediction[allergens[allergen]] = predict_allergen(image_data, allergen)
        print(f"Done {allergen+1}/len(allergens)")

    return prediction
# ---------------------- SCRIPT END -----------------------

@app.route('/api/process-image', methods=['POST'])
def predict_allergens_endpoint():
    try:
        # Get the image data from the request (assuming it's sent as raw bytes)
        # data = request.json
        # image_data = data['imageData']
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

if __name__ == '__main__':
     app.run(debug=True)  # Run the app in debug mode