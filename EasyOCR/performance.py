import easyocr
import PIL
from PIL import Image, ImageDraw

import pandas as pd

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

# --------------------------------------------------
reader = easyocr.Reader(['en'], gpu=False)  # English
# --------------------------------------------------

def predict_allergens(image_path):
    # ------------ Extract text ---------------------
    image_path = 'CLIP/feature_engineering_dataset/enhanced_data/002.jpg'
    result = reader.readtext(image_path)
    text = ' '.join(item[1] for item in result)
    # -----------------------------------------------

    prediction_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # SEARCH IN THE TEXT IF THE ALLERGENS ARE PRESENT, AND UPDATE THE PREDICTION ARRAY
    # BY SETTING THE VALUE 1 WHEN THE ALLERGEN IS PRESENT
    # THE ORDER OF THE ALLERGENS IS WRITTEN LINE 7
    # 
    # SEARCH COULD BE DONE IN TWO WAYS:
    #       - REGULAR EXPRESSIONS: SEARCH IF KEYWORDS (Example: gluten or wheat for gluten)
    #                           ARE PRESENT IN THE TEXT.
    #       - USE A MODEL: SEARCH FOR SOME MODEL THAT COULD RECOGNIZE SOME WORDS WITHIN A TEXT




def measure_performance():
    # -------------- Read Excel labellisation file ----------
    excel_file_path = 'labellisation.xlsx'
    df = pd.read_excel(excel_file_path)
    excel_col_range = [1, 15]
    excel_row_shift = 1
    # -------------------------------------------------------


    # -------------- Image Range declaration ----------------
    directory_path = "feature_engineering_dataset/raw_data/"
    first_image = 1     # 001.jpg
    last_image = 20     # 020.jpg
    extension = ".jpg"
    # -------------------------------------------------------


    for image in range(first_image, last_image + 1):  # from 001.jpg to 02O.jpg
        true_array = df.iloc[image+excel_row_shift, excel_col_range[0]:excel_col_range[1]].values
        prediction_array = predict_allergens(f"{directory_path}{image:03d}{extension}")

# ----------- CALCULATE MANY METRICS -----------------
        # CALCULATE THE NUMBER OF TRUE POSITIVES, TRUE NEGATIVES, FALSE POSITIVES and FALSE NEGATIVES
        # AND CALCULATE METRICS


        # THE CODE BELOW WAS USED FOR CALCULATING ACCURACY, BUT IT WOULD BE NICE TO COMPUTE
        # MORE ADVANCED METRICS
        # nb_allergens = len(allergens)
        # score = 0
        # for i in range(nb_allergens):
        #     score += 1 - abs(true_array[i] - prediction_array[i])
        # score = score / nb_allergens
    # detailed_scores.append(score)
# --------------------------------------------------
        
        print(f"{image}/{last_image - first_image + 1} images processed.")

    # RETURN ALL METRICS


def main():
    performance_metrics = measure_performance()

    # ANALYSE PERFORMANCE METRICS AND DRAW SOME GRAPHS FOR EXAMPLE

if __name__ == "__main__":
    print("librairies sccessfully imported.")
    main()