from PIL import Image
import requests

import pandas as pd

from statistics import mean

from transformers import CLIPProcessor, CLIPModel

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

nb_allergens = len(allergens)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def predict(image, allergen):
    image = Image.open(image)
    match allergen:
        case 0:
            inputs = processor(text=["list of ingredients containing the word 'oat', 'barley', 'wheat', 'rye', 'wheat' or 'gluten'", "list of ingredients containing neither the word 'oat', the word 'barley', the word 'wheat', the word 'rye', the word 'wheat' nor the word 'gluten'"], images=image, return_tensors="pt", padding=True)
        case 1:
            inputs = processor(text=["list of ingredients containing the word 'egg'", "list of ingredients not containing the word 'egg'"], images=image, return_tensors="pt", padding=True)
        case 2:
            inputs = processor(text=["list of ingredients containing the word 'milk'", "list of ingredients not containing the word 'milk'"], images=image, return_tensors="pt", padding=True)
        case 3:
            inputs = processor(text=["list of ingredients containing the word 'almond', 'hazelnut', 'walnuts' or 'nut'", "list of ingredients containing neither the word 'almond', the word 'hazelnut', the word 'walnuts' nor the word'nut'"], images=image, return_tensors="pt", padding=True)
        case 4:
            inputs = processor(text=["list of ingredients containing the word 'peanuts'", "list of ingredients not containing the word 'peanuts"], images=image, return_tensors="pt", padding=True)
        case 5:
            inputs = processor(text=["list of ingredients containing the word 'soy'", "list of ingredients not containing the word 'soy'"], images=image, return_tensors="pt", padding=True)
        case 6:
            inputs = processor(text=["list of ingredients containing the word 'mollusc', 'mussel', 'snail', 'squid', 'oyster' or 'whelk'", "list of ingredients containing neither the word 'mollusc', the word 'mussel', the word 'snail', the word 'squid', the word 'oyster' nor the word'whelk'"], images=image, return_tensors="pt", padding=True)
        case 7:
            inputs = processor(text=["list of ingredients containing the word 'fish'", "list of ingredients not containing the word 'fish'"], images=image, return_tensors="pt", padding=True)
        case 8:
            inputs = processor(text=["list of ingredients containing the word 'lupin'", "list of ingredients not containing the word 'lupin'"], images=image, return_tensors="pt", padding=True)
        case 9:
            inputs = processor(text=["list of ingredients containing the word 'crustaceans', 'crab', 'lobster', 'prawn', 'shrimp' or 'scampi'", "list of ingredients containg neither the word 'crustaceans', the word 'crab', the word 'lobster', the word 'prawn', the word 'shrimp' nor the word 'scampi'"], images=image, return_tensors="pt", padding=True)
        case 10:
            inputs = processor(text=["list of ingredients containing the word 'sesame'", "list of ingredients not containing the word 'sesame'"], images=image, return_tensors="pt", padding=True)
        case 11:
            inputs = processor(text=["list of ingredients containing the word 'mustard'", "list of ingredients containing the word 'mustard'"], images=image, return_tensors="pt", padding=True)
        case 12:
            inputs = processor(text=["list of ingredients containing the word 'celery'", "list of ingredients not containing the word 'celery'"], images=image, return_tensors="pt", padding=True)
        case 13:
            inputs = processor(text=["list of ingredients containing the word 'sulphite'", "list of ingredients not containing the word 'sulphite'"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    
    return probs[0][0].item()


def main():
    excel_file_path = 'labellisation.xlsx'
    df = pd.read_excel(excel_file_path)
    excel_col_range = [1, 15]
    excel_row_shift = 1

    detailed_scores = []

    for image in range(1, 21):  # 
        true_array = df.iloc[image+excel_row_shift, excel_col_range[0]:excel_col_range[1]].values
        prediction_array = []
        for allergen in range(0, 14):
            prediction_array.append(predict(f"feature_engineering_dataset/raw_data/{image:03d}.jpg", allergen))    # First arg: image path; Second: allergen (0: gluten, 1: eggs...). See allergens line 6.
        
        score = 0
        for i in range(nb_allergens):
            score += 1 - abs(true_array[i] - prediction_array[i])
        score = score / nb_allergens

        detailed_scores.append(score)
        print(f"{image}/20 images processed.")
    
    total_score = mean(detailed_scores)

    print(f"Score: {round(total_score*100, 1)}%")



if __name__ == "__main__":
    print("librairies sccessfully imported.")
    main()