from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import cv2

# prepare image + question
pic = "man_with_tie.jpg"
image = cv2.imread(pic)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
text = "what color is the trousers?"



processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
