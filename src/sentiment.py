'''
Script containing functions to load and use roberta pretrained models.
'''

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, CamembertTokenizer


def load_classification_model_hugging_face(
        model_name="cardiffnlp/twitter-roberta-base-sentiment",
        device = -1,
        max_l = 512,
        batch_size_sentiment = 1
    ):
    '''
    Load sentiment classification model from hugging face
    '''

    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
        max_length = max_l,
        truncation = True,
        batch_size = batch_size_sentiment
    )
    
    return classifier

def classify_sentiment_text_hugging_face(text, classification_model, dictionary_labels = None):
    '''
    Classify a text and return the results
    '''

    # Classify all texts.
    result = classification_model(text)

    labels = []
    scores = []

    # If there are dictionary labels, loop through the list and asign each label to its equivalent.
    for entry in result:
        if dictionary_labels is not None:
            entry['label'] = dictionary_labels[entry['label']]
        labels.append(entry['label'])
        scores.append(entry['score'])

    return labels, scores