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
                    clean_html_text = True, perform_sentiment_classification = True, sent_device = -1, labels = None, max_length_data = 512,
                    batch_size = 1):
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
        model = load_classification_model_hugging_face(
            model_name=m_type,
            device=sent_device,
            max_l = max_length_data,
            batch_size_sentiment=batch_size
        )

        def classify_texts_in_chunks(texts_list):
            
            # Divide texts in chunks and save the number of chunks for each text.
            group_sizes = []
            decoded_chunks_list = []
            for text in texts_list:
                tokenized_text = tokenizer(
                    text,
                    max_length=divide_in_chunks,
                    truncation = True,
                    padding = True,
                    return_overflowing_tokens=True,
                    stride=128,
                    return_tensors='pt'
                )["input_ids"]

                # Compute and save number of chunks
                group_sizes.append(len(tokenized_text))
                # Decode chunks
                decoded_chunks = tokenizer.batch_decode(tokenized_text, skip_special_tokens=True)
                # Save all chunks into a single non-nested list.
                for c in decoded_chunks:
                    decoded_chunks_list.append(c)

            # Perform sentiment classification for all chunks equally
            sentiment, scores_classification = classify_sentiment_text_hugging_face(
                text = decoded_chunks_list,
                classification_model = model,
                dictionary_labels = labels
            )

            # Iterate in groups, use group_sizes list to rebuild chunks and compute their predominant emotion and mean score.
            start = 0
            results_emotions = []
            results_scores = []
            for size in group_sizes:
                sentiments_text = sentiment[start:start + size]
                scores_text = scores_classification[start:start + size]

                # Keep the ammount of chunks assigned to each emotion with their corresponding score.
                labels_dictionary_chunks = {}
                scores = {}

                for sent, score in zip(sentiments_text, scores_text):
                    # Keep results in respective dictionaries
                    if sent in labels_dictionary_chunks:
                        labels_dictionary_chunks[sent] += 1
                        scores[sent].append(score)
                    else:
                        labels_dictionary_chunks[sent] = 1
                        scores[sent] = [score]

                # Select the predominant and compute the average score for set emotion.

                max_value = max(labels_dictionary_chunks.values())
                predominant_emotions = [k for k, v in labels_dictionary_chunks.items() if v == max_value]
                final_mean_scores = []
                
                for em in predominant_emotions:
                    final_mean_scores.append(statistics.mean(scores[em]))
                
                results_emotions.append('-'.join(predominant_emotions))
                results_scores.append(final_mean_scores)

                start += size
            
            return results_emotions, results_scores

        # Classify texts
        
        if divide_in_chunks is not None:
            # Loading appropiate tokenizer just to check if the length of the text to classify exceeds 512.
            tokenizer = AutoTokenizer.from_pretrained(m_type, use_fast=True)

            # Classify texts into emotions.
            sentiment_label, scores_classification = classify_texts_in_chunks(
                texts_list=data[text_column].tolist()
            )

            data["emotion"] = sentiment_label
            data["emotion_score"] = scores_classification

        else:
            # Process all texts using the user specified batch size.
            sentiment_label, scores_classification = classify_sentiment_text_hugging_face(
                text = data[text_column].tolist(),
                classification_model=model,
                dictionary_labels=labels
            )

            data["emotion"] = sentiment_label
            data["emotion_score"] = scores_classification

            
            #data["review_emotion"] = data[text_column].apply(classify_sentiments)

        # Adding emotion and emotion_score to the columns that must be selected
        columns_selected.extend(["emotion", "emotion_score"])

    # Renaming text column so it its name can be used in other functions of the pipeline
    data = data.rename(columns = {text_column: "text"})

    # Selecting only columns of interest without duplicates
    data = data[columns_selected + columns_to_keep]

    return data