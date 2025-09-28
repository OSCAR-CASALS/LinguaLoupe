'''
This script contains the function necessary to process texts and classify them based on wether they are positive, neutral or negative.
'''
import os
from pathlib import Path

from transformers import AutoTokenizer
import statistics
import pandas as pd
from src.sentiment import load_roberta_classification_model, classify_text_sentiment
import swifter
import warnings

def process_reviews(data_path, text_column, csv_sep = ",",
                    min_rows_to_parallelize = 10000, cancel_parallelisation = False, columns_to_keep = [],
                    convert_to_string = False, divide_in_chunks = 512):
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

    # Checking the product description of each product is in string format
    if pd.api.types.infer_dtype(data[text_column]) != "string":
        if convert_to_string == True:
            data[text_column] = data[text_column].astype(str)
        else:
            raise TypeError("The text of each review must be in string format.")

    # loading model
    model = load_roberta_classification_model()

    # Determining if parallelization is required or not
    use_swifter = False
    if (data.shape[0] >= min_rows_to_parallelize) and (cancel_parallelisation == False):
        use_swifter = True

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", use_fast=True)
    def classify_sentiments(text):
        '''
        Function that performs Sentiment classification.
        '''
        
        # If the size of the text is bigger than what ROBERTA can take,
        # split it.
        tokenized_text = tokenizer(text)["input_ids"]
        t_size = len(tokenized_text)
        if (t_size > 512) and (divide_in_chunks is not None):
            labels_to_numbers = {
                "NEGATIVE": 0,
                "NEUTRAL": 1,
                "POSITIVE": 2
            }
            # To know which classification has been selected the most, a list with 3 zeros has been created, each
            # corresponding to a sentiment (negative, neutral and positive). Everytime a classification is made,
            # 1 is added to the corresponding indes. Once all classifications have been made, it is just a mater of selecting the biggest
            # number
            sents = [0, 0, 0]
            scores = [[], [], []]
            c = 0
            for i in range(0, len(text), divide_in_chunks):
                if (i + divide_in_chunks) < (len(text) - 1):
                    classification = classify_text_sentiment(text[i:i + divide_in_chunks], model)
                    index_list = labels_to_numbers[classification["label"]]
                    sents[index_list] += 1
                    scores[index_list].append(classification["score"])
                    c += 1
            if c > 1:
                warnings.warn(f"Found a text with more than 512 tokens ({t_size} tokens), the text will be divided into chunks of {divide_in_chunks}. After classifiying each chunk the predominant emotion will be selected.")
            # Finding predominant emotion
            max_sentiment = max(sents)
            chosen_sentiments = []
            for i in range(0, len(sents)):
                if sents[i] == max_sentiment:
                    chosen_sentiments.append(i)
            # Computing and returning result, the reason why it is a list of list and not a single list
            # is in case of draws between emotions.
            result_sents = []
            result_scores = []
            for i in chosen_sentiments:
                result_sents.append(list(labels_to_numbers.keys())[i])
                result_scores.append(statistics.mean(scores[i]))
            return ['-'.join(result_sents), result_scores]
        # If the text has less than 512 tokens, just classify the whole text with Roberta.
        # The scores are returned in a list for consistency with the results of text with
        # a high ammount of tokens.
        sentiment = classify_text_sentiment(text, model)
        return [sentiment["label"], [sentiment["score"]]]

    # Use swifter to classify all descriptions using multithreading or just do an apply in case
    # the data does not have many rows.
    if use_swifter == True:
        data["review_emotion"] = data[text_column].swifter.apply(classify_sentiments)
    else:
        data["review_emotion"] = data[text_column].apply(classify_sentiments)

    # Dividing review emotion into two columns, one with the label assifgned by classify_text_sentiment and the
    # other with the score assigned to the classification.

    data["emotion"] = data["review_emotion"].str[0]
    data["emotion_score"] = data["review_emotion"].str[1]

    # Removing column review_emotion since it is redundant.
    data.drop("review_emotion", axis=1, inplace=True)

    # Renaming text column so it its name can be used in other functions of the pipeline
    data = data.rename(columns = {text_column: "text"})

    # Selecting only columns of interest
    data = data[["text","emotion", "emotion_score"] + columns_to_keep]

    return data