'''
This script contains the function necessary to process texts and classify them based on wether they are positive, neutral or negative.
'''
import os
from pathlib import Path

from transformers import AutoTokenizer
import statistics
import pandas as pd
from src.sentiment import load_classification_model_hugging_face, classify_sentiment_text_hugging_face
import warnings

from bs4 import BeautifulSoup

def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def process_reviews(data_path, text_column, csv_sep = ",",
                    columns_to_keep = [],
                    convert_to_string = False, divide_in_chunks = 512, language = "english", m_type="social_media",
                    clean_html_text = True, perform_sentiment_classification = True, sent_device = -1, labels = None):
    '''
    A function in charge of classifiying texts into positive, negative, or neutral.
    '''
     # Checking if data is in dataframe format or instead is a path to a file or url
    if isinstance(data_path, str):
        # Geting file sufix to determine how to import data to python
        file = Path(data_path)
        # Reading data
        match file.suffixes[0]:
            case ".jsonl":
                data = pd.read_json(data_path, lines=True, compression="infer")
            case ".json":
                data = pd.read_json(data_path, lines=False, compression="infer")
            case ".csv":
                data = pd.read_csv(data_path, compression="infer", sep=csv_sep)
            case ".xlsx":
                if len(file.suffixes) > 1:
                    if file.suffixes[-1] != ".zip":
                        raise Exception("Excel files can only be .zip compressed or uncompressed.")
                data = pd.read_excel(data_path)
            case ".tsv":
                data = pd.read_csv(data_path, compression="infer", sep="\t")
            case _:
                raise Exception("Only the following file formats are allowed: jsonl, json, csv, xlsx, tsv")
    elif isinstance(data_path, pd.DataFrame):
        data = data_path.copy()
    else:
        raise TypeError("data_path can only be a string or a pandas dataframe")

    # Checking the text column is in string format
    if pd.api.types.infer_dtype(data[text_column]) != "string":
        if convert_to_string == True:
            data[text_column] = data[text_column].astype(str)
        else:
            raise TypeError("The text of each review must be in string format.")
        
    # Cleaning html from text column unless specified otherwise
    if clean_html_text == True:
        data[text_column] = data[text_column].apply(clean_html)

    # Removing texts that do not contain at least one alphabetic character, or are NA.
    data = data[data[text_column].str.contains(r"[A-Za-z]", na=False)]

    # If dataset already contains a column called emotion, changed it to emotion_original so it does not interfere with the pipeline (to find a better fix later).
    
    if "emotion" in data.columns.tolist():
        data = data.rename(columns = {"emotion": "emotion_original"})
        if "emotion" in columns_to_keep:
            columns_to_keep.append("emotion_original")
    if "emotion_score" in data.columns.tolist():
        data = data.rename(columns = {"emotion_score": "emotion_score_original"})
        if "emotion_score" in columns_to_keep:
            columns_to_keep.append("emotion_score_original")

    # If the perform_sentiment_classification is true, perform sentiment classification, otherwise just rename text column
    columns_selected = ["text"]
    if perform_sentiment_classification == True:
        # Removing emotion column in columns to keep just in case so we don't have duplicate columns
        if "emotion" in columns_to_keep:
            columns_to_keep.remove("emotion")
        if "emotion_score" in columns_to_keep:
            columns_to_keep.remove("emotion_score")
            
        # loading model
        model = load_classification_model_hugging_face(model_name=m_type, device=sent_device)

        # Loading appropiate tokenizer just to check if the length of the text to classify exceeds 512.
        if divide_in_chunks is not None:
            tokenizer = AutoTokenizer.from_pretrained(m_type, use_fast=True)
            

        # Classify texts

        def classify_sentiments(text):
            '''
            Function that performs Sentiment classification.
            '''

            if divide_in_chunks is not None:
                # If the size of the text is bigger than what ROBERTA can take,
                # split it.
                tokenized_text = tokenizer(
                    text,
                    max_length=divide_in_chunks,
                    truncation = True,
                    padding = True,
                    return_overflowing_tokens=True,
                    stride=128,
                    return_tensors='pt'
                )["input_ids"]

                t_size = len(tokenized_text)
                
                if (t_size > 1):

                    #Apply classification to each fragment of the text divided in chunks

                    labels_dictionary_chunks = {}
                    scores = {}

                    for chunk_ids in tokenized_text:
                        iter_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                        sentiment = classify_sentiment_text_hugging_face(
                            text = iter_text,
                            classification_model = model,
                            dictionary_labels = labels
                            )

                        # Keep results in respective dictionaries
                        sent = sentiment["label"]

                        if sent in labels_dictionary_chunks:
                            labels_dictionary_chunks[sent] += 1
                            scores[sent].append(sentiment["score"])
                        else:
                            labels_dictionary_chunks[sent] = 1
                            scores[sent] = [sentiment["score"]]

                    max_value = max(labels_dictionary_chunks.values())
                    predominant_emotions = [k for k, v in labels_dictionary_chunks.items() if v == max_value]
                    final_mean_scores = []

                    for em in predominant_emotions:
                        final_mean_scores.append(statistics.mean(scores[em]))

                    return ['-'.join(predominant_emotions), final_mean_scores]

                    
            # If the text has less than 512 tokens, just classify the whole text with Roberta.
            # The scores are returned in a list for consistency with the results of text with
            # a high ammount of tokens.
            sentiment = classify_sentiment_text_hugging_face(
                text = text,
                classification_model=model,
                dictionary_labels=labels
            )

            return [sentiment["label"], [sentiment["score"]]]

        # Classify texts into emotions.
        data["review_emotion"] = data[text_column].apply(classify_sentiments)

        # Dividing review emotion into two columns, one with the label assifgned by classify_text_sentiment and the
        # other with the score assigned to the classification.

        data["emotion"] = data["review_emotion"].str[0]
        data["emotion_score"] = data["review_emotion"].str[1]

        # Removing column review_emotion since it is redundant.
        data.drop("review_emotion", axis=1, inplace=True)

        # Adding emotion and emotion_score to the columns that must be selected
        columns_selected.extend(["emotion", "emotion_score"])

    # Renaming text column so it its name can be used in other functions of the pipeline
    data = data.rename(columns = {text_column: "text"})

    # Selecting only columns of interest without duplicates
    data = data[columns_selected + columns_to_keep]

    return data