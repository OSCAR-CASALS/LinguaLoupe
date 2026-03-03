'''
Script containing functions to load and use roberta pretrained models.
'''

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, CamembertTokenizer
from pysentimiento import create_analyzer

def load_classification_model(language = "english", model_type="social_media"):
    '''
    Load classification model Roberta
    '''
    
    if model_type == "social_media":
        # Models to use when analyzing social media text
        if language == "english":
            model_name = "cardiffnlp/twitter-roberta-base-sentiment"
            #model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, max_length=512, truncation=True)

            return classifier
        
        elif language == "spanish":
            analyzer = create_analyzer(task="sentiment", lang="es")
            return analyzer
        
    else:
        if language == "english":
            classifier = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
            return classifier
    
    classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
    return classifier
    #classifier = pipeline("sentiment-analysis", model=model_name, tokenizer = model_name)


def classify_text_sentiment(text, classification_model, model_type="social_media"):
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
    if model_type == "social_media":
        return {
            "label": labels[result["label"]],
            "score": result["score"]
        }
    
    return {
            "label": result["label"],
            "score": result["score"]
        }

def classify_text_no_english(text, model, language="spanish", model_type="social_media"):
    '''
    Perform sentiment classification using a Pysentimiento or bert-base-multilingual-uncased-sentiment model.
    '''

    if language == "spanish" and model_type == "social_media":
        sentiment = model.predict(text)
            
        labels = {
            "NEG": "NEGATIVE",
            "NEU": "NEUTRAL",
            "POS": "POSITIVE"
        }

        return {
            "label" : labels[sentiment.output],
            "score": sentiment.probas[sentiment.output]
        }
    
    sentiment = model(text)[0]

    stars = int(sentiment['label'].split(' ')[0])
    label = ""

    if stars > 3:
        label = "POSITIVE"
    elif stars < 3:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "label": label,
        "score": sentiment["score"]
    }