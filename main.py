from transformers import ViltProcessor, ViltForQuestionAnswering
import cv2
import pandas as pd



# questions will be answerd by the model
questions = ['is the person a male or a female?' , 'is the person wearing a hat?'
            , 'what color is the shirt?', 'what color is the trouser or the skirt or the short?',
            'is the person wearing a tie?','is the person wearing a trouser?'
            , 'is the person wearing a glasses?' , 'is the person wearing a short?',
            'is the person fat?']



maped_questions = ['gender', 'hat', 'shirt_color', 'trouser_color', 'tie', 'trouser', 'glasses', 'short', 'fat']

# load model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# load image
img = cv2.imread('fat_man.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

answer = []
df = pd.DataFrame(columns=['question', 'answer'])

# loop over questions
for question in questions:
    # prepare inputs
    questions_maped = maped_questions[questions.index(question)]
    encoding = processor(img, question, return_tensors="pt")
    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    # print("Predicted answer:", model.config.id2label[idx])
    answer.append(model.config.id2label[idx])
    df = df.append({'question': questions_maped, 'answer': model.config.id2label[idx]}, ignore_index=True)

print('the answer is ',answer)
print(df)