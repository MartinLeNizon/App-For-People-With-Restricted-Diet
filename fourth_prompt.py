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
            inputs = processor(text=["ingredient list of a product containing wheat or gluten", "ingredient list of a product without wheat nor gluten"], images=image, return_tensors="pt", padding=True)
        case 1:
            inputs = processor(text=["ingredient list of a product containing eggs", "ingredient list of a product without eggs"], images=image, return_tensors="pt", padding=True)
        case 2:
            inputs = processor(text=["ingredient list of a product containing milk", "ingredient list of a product without milk"], images=image, return_tensors="pt", padding=True)
        case 3:
            inputs = processor(text=["ingredient list of a product containing nuts", "ingredient list of a product without nuts"], images=image, return_tensors="pt", padding=True)
        case 4:
            inputs = processor(text=["ingredient list of a product containing peanuts", "ingredient list of a product without peanuts"], images=image, return_tensors="pt", padding=True)
        case 5:
            inputs = processor(text=["ingredient list of a product containing soya", "ingredient list of a product without soya"], images=image, return_tensors="pt", padding=True)
        case 6:
            inputs = processor(text=["ingredient list of a product containing molluscs", "ingredient list of a product without molluscs"], images=image, return_tensors="pt", padding=True)
        case 7:
            inputs = processor(text=["ingredient list of a product containing fish", "ingredient list of a product without fish"], images=image, return_tensors="pt", padding=True)
        case 8:
            inputs = processor(text=["ingredient list of a product containing lupin", "ingredient list of a product without lupin"], images=image, return_tensors="pt", padding=True)
        case 9:
            inputs = processor(text=["ingredient list of a product containing crustaceans", "ingredient list of a product without crustaceans"], images=image, return_tensors="pt", padding=True)
        case 10:
            inputs = processor(text=["ingredient list of a product containing sesame", "ingredient list of a product without sesame"], images=image, return_tensors="pt", padding=True)
        case 11:
            inputs = processor(text=["ingredient list of a product containing mustard", "ingredient list of a product without mustard"], images=image, return_tensors="pt", padding=True)
        case 12:
            inputs = processor(text=["ingredient list of a product containing celery", "ingredient list of a product without celery"], images=image, return_tensors="pt", padding=True)
        case 13:
            inputs = processor(text=["ingredient list of a product containing sulphites", "ingredient list of a product without sulphites"], images=image, return_tensors="pt", padding=True)

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