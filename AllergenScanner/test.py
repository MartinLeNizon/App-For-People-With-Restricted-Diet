# ---------------------- SCRIPT -----------------------
import base64
import requests

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
    
    # Getting the base64 string
    base64_image = encode_image("AllergenScanner/001.jpg")
    # base64_image = image_data

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
                "url": f"data:image/jpeg;base64,{base64_image}",
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
    
    if content == "Yes":
        return True
    elif content == "No":
        return False
    else:
        print("ERROR: VISION result not formatted")

def predict_all_allergens(image_data):
    prediction = {}  # Initialize an empty dictionary
    for allergen in range(len(allergens)):
        prediction[allergens[allergen]] = predict_allergen(image_data, allergen)

    return prediction
# ---------------------- SCRIPT END -----------------------

print(predict_all_allergens(0))