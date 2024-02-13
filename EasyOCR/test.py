import easyocr
import PIL
from PIL import Image, ImageDraw

reader = easyocr.Reader(['en'], gpu=False)  # English

image_path = 'CLIP/feature_engineering_dataset/enhanced_data/002.jpg'
result = reader.readtext(image_path)

text = ' '.join(item[1] for item in result)

print(text)