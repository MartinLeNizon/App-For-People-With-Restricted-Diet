from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("App-For-People-With-Restricted-Diet/dataset/003.jpg")

inputs = processor(text=["a product containing wheat or gluten", "a product without wheat nor gluten"], images=image, return_tensors="pt", padding=True)
# inputs = processor(text=["a product containing eggs", "a product without eggs"], images=image, return_tensors="pt", padding=True)
# inputs = processor(text=["a product containing milk", "a product without milk"], images=image, return_tensors="pt", padding=True)
# inputs = processor(text=["a product containing nuts", "a product without nuts"], images=image, return_tensors="pt", padding=True)
# inputs = processor(text=["a product containing peanuts", "a product without peanuts"], images=image, return_tensors="pt", padding=True)
# inputs = processor(text=["a product containing soya", "a product without soya"], images=image, return_tensors="pt", padding=True)
# inputs = processor(text=["a product containing molluscs", "a product without molluscs"], images=image, return_tensors="pt", padding=True)
# inputs = processor(text=["a product containing fish", "a product without fish"], images=image, return_tensors="pt", padding=True)
# inputs = processor(text=["a product containing lupin", "a product without lupin"], images=image, return_tensors="pt", padding=True)
# inputs = processor(text=["a product containing crustaceans", "a product without crustaceans"], images=image, return_tensors="pt", padding=True)
# inputs = processor(text=["a product containing sesame", "a product without sesame"], images=image, return_tensors="pt", padding=True)
# inputs = processor(text=["a product containing mustard", "a product without mustard"], images=image, return_tensors="pt", padding=True)
# inputs = processor(text=["a product containing celery", "a product without celery"], images=image, return_tensors="pt", padding=True)
# inputs = processor(text=["a product containing sulphites", "a product without sulphites"], images=image, return_tensors="pt", padding=True)



outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(logits_per_image)
print(probs)