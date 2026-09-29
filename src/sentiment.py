'''
Script containing functions to load and use roberta pretrained models.
'''

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, CamembertTokenizer
from pysentimiento import create_analyzer


def load_classification_model_hugging_face(
        model_name="cardiffnlp/twitter-roberta-base-sentiment",
        device = -1
    ):
    '''
    Load sentiment classification model from hugging face
    '''

    classifier = pipeline("sentiment-analysis", model=model_name, device=device)
    
    return classifier

def classify_sentiment_text_hugging_face(text, classification_model, dictionary_labels = None):
    '''
    Classify a text and return the results
    '''

    result = classification_model(text)[0]

    if dictionary_labels is not None:
        return {
            'label': dictionary_labels[result['label']],
            'score': result['score']
        }

    return {
        'label': result['label'],
        'score': result['score']
    }