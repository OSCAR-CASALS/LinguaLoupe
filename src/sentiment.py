'''
Script containing functions to load and use roberta pretrained models.
'''

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, CamembertTokenizer
from pysentimiento import create_analyzer

def load_roberta_classification_model(language = "english"):
    '''
    Load classification model Roberta
    '''
    
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
    elif language == "italian":
        analyzer = create_analyzer(task="sentiment", lang="it")
        return analyzer
    elif language == "portuguese":
        analyzer = create_analyzer(task="sentiment", lang="pt")
        return analyzer
    #classifier = pipeline("sentiment-analysis", model=model_name, tokenizer = model_name)


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

def classify_text_pysentimiento(text, pysentimiento_model, language="spanish"):
    '''
    Perform sentiment classification using a Pysentimiento instances
    '''

    sentiment = pysentimiento_model.predict(text)

    if language == "italian":

        labels = {
            "neg": "NEGATIVE",
            "pos": "POSITIVE"
        }

        sent = sentiment.output

        if len(sent) == 0:
            sent = "NEUTRAL"

            sc = 1 - sum(sentiment.probas.values())
            
            return {
                "label" : sent,
                "score": sc
            }
        
        elif len(sent) > 1:
            sent = "MIXED"

            sc = sum(sentiment.probas.values())/2

            return {
                "label" : sent,
                "score": sc
            }
        else:
            sent = labels[sent[0]]

            return {
                "label" : sent,
                "score": sentiment.probas[sentiment.output[0]]
            }
        
    labels = {
        "NEG": "NEGATIVE",
        "NEU": "NEUTRAL",
        "POS": "POSITIVE"
    }

    return {
        "label" : labels[sentiment.output],
        "score": sentiment.probas[sentiment.output]
    }