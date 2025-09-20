'''
Script containing functions to load and use roberta pretrained models.
'''

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def load_roberta_classification_model():
    '''
    Load classification model Roberta
    '''
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, max_length=512, truncation=True)

    return classifier


def classify_text_sentiment(text, classification_model):
    '''
    Use Roberta to classify the sentiments of a text into POSITIVE, NEUTRAL, and NEGATIVE.
    '''
    
    # Dictionary with posible labels
    labels = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE"
    }

    # Apliying ROBERTA model

    result = classification_model(text)[0]

    # Asigning the corresponding label to key "label" in the dictionary.

    return {
        "label": labels[result["label"]],
        "score": result["score"]
    }