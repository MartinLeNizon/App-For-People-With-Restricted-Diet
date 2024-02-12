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
            inputs = processor(text=["gluten-containing product's ingredient list", "gluten-free product's ingredient list"], images=image, return_tensors="pt", padding=True)
        case 1:
            inputs = processor(text=["egg-containing product's ingredient list", "egg-free product's ingredient list"], images=image, return_tensors="pt", padding=True)
        case 2:
            inputs = processor(text=["milk-containing product's ingredient list'", "milk-free product's ingredient list"], images=image, return_tensors="pt", padding=True)
        case 3:
            inputs = processor(text=["nut-containing product's ingredient list", "nut-free product's ingredient list"], images=image, return_tensors="pt", padding=True)
        case 4:
            inputs = processor(text=["peanut-containing product's ingredient list", "peanut-free product's ingredient list"], images=image, return_tensors="pt", padding=True)
        case 5:
            inputs = processor(text=["soy-containing product's ingredient list", "soy-free product's ingredient list"], images=image, return_tensors="pt", padding=True)
        case 6:
            inputs = processor(text=["molluscs-containing product's ingredient list", "molluscs-free product's ingredient list"], images=image, return_tensors="pt", padding=True)
        case 7:
            inputs = processor(text=["fish-containing product's ingredient list", "fish-free product's ingredient list"], images=image, return_tensors="pt", padding=True)
        case 8:
            inputs = processor(text=["lupin-containing product's ingredient list", "lupin-free product's ingredient list"], images=image, return_tensors="pt", padding=True)
        case 9:
            inputs = processor(text=["crustaceans-containing product's ingredient list", "crustaceans-free product's ingredient list"], images=image, return_tensors="pt", padding=True)
        case 10:
            inputs = processor(text=["sesame-containing product's ingredient list", "sesame-free product's ingredient list"], images=image, return_tensors="pt", padding=True)
        case 11:
            inputs = processor(text=["mustard-containing product's ingredient list", "mustard-free product's ingredient list"], images=image, return_tensors="pt", padding=True)
        case 12:
            inputs = processor(text=["celery-containing product's ingredient list", "celery-free product's ingredient list"], images=image, return_tensors="pt", padding=True)
        case 13:
            inputs = processor(text=["sulphite-containing product's ingredient list", "sulphite-free product's ingredient list"], images=image, return_tensors="pt", padding=True)

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