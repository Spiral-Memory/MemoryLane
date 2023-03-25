from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
# Load the trained model

tags = ['Greetings','Name Inquiry','Address Inquiry','Last meeting Inquiry','Relationship Inquiry','GoodBye']
model = DistilBertForSequenceClassification.from_pretrained("Projects\Intent_HF\intent_cf_model")

# Define the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model.eval()

with torch.no_grad():
    while True:

        text = input("Enter the text to be evaluated: ")
        inputs = tokenizer(text,padding=True,truncation=True,max_length=512,return_tensors="pt")

        # Pass the tokenized text to the model to get the predictions
        outputs = model(**inputs)
        predictions = F.softmax(outputs['logits'], dim=1)
        print(predictions)
        max_score, max_index = torch.max(predictions[0], dim=0)

        if max_score > 0.95:
            print(tags[max_index])
        else:
            print("I am not sure what you are asking.")
