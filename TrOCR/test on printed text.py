from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image

# load image from the IAM dataset
url = 'https://ih1.redbubble.net/image.335620159.9954/flat,750x,075,f-pad,750x1000,f8f8f8.u7.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)
