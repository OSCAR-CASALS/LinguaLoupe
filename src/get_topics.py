'''
Functions for topic analysis.
'''

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import pandas as pd
import warnings
def load_BERT(lang = "english", min_topic_size=10):
    '''
    Creates a BERTtopic model using topic representation KeyBERTInspired.
    '''
    representation_model = KeyBERTInspired()
    return BERTopic(language=lang, verbose=True, representation_model=representation_model, min_topic_size=min_topic_size)

def get_topics(model, df, reviews_columns):
    '''
    Uses a BERTtopic model to find topics in a dataframe with texts.
    '''
    docs = df[reviews_columns].to_list()
    return model.fit_transform(docs)

def topic_modelling(df, review_columns, min_topic_size=10, language="english"):
    '''
    Classifies reviews in different topics.
    '''
    # Loading model and dividing in topics
    topic_model = load_BERT(min_topic_size=min_topic_size, lang=language)
    topic, probs = get_topics(topic_model, df, review_columns)

    # Adding the topic number and the probability of belonging to se topic to each review.
    df["topic"] = topic
    df["probability_topic"] = probs

    # Geting the 10 most frequent topics.
    top_topics = topic_model.get_topic_freq()
    # removing outliyer topic
    top_topics = top_topics[top_topics["Topic"] != -1]
    # If after removing outliyers there are no topics lefy in top topics. then no topics have been selected and the function will try again with 
    # half the topic size
    umap_failed = False
    try:
        topic_model.visualize_topics()
    except Exception as e:
        umap_failed = True
    if (top_topics.shape[0] == 0) or (umap_failed == True):
        reduced_topic_size = int(min_topic_size/2)
        if reduced_topic_size >= 2:
            warnings.warn(f"No topics identified for the dataframe, triying again reducing by half the min_topic_size({reduced_topic_size})")
            return topic_modelling(df, review_columns, reduced_topic_size, language=language)
        warnings.warn("Could not find topics for the dataframe")
    # Getting most important words for each topic
    main_words = []
    score = []

    def add_words(x):
        t = topic_model.get_topic(x)
        w = []
        s = []
        for i in t:
            w.append(i[0])
            s.append(str(i[1]))
        
        main_words.append(','.join(w))
        score.append(','.join(s))

    top_topics["Topic"].apply(add_words)

    top_topics["Main Words"] = main_words

    top_topics["c-TF-IDF score"] = score

    # Returning model and dataframe with top 10 topics

    return topic_model, top_topics

def review_topics(df, review_column = "text",emotion_column = "emotion", min_topic_size=10, language="english"):
    '''
    Divide positive, neutral and negative texts into topics.
    '''
    # Defining variables where results will be kept
    positive_results = []
    neutral_results = []
    negative_results = []
    neg_net_results = []
    neg_pos_results = []
    net_pos_results = []
    all_results = []

    # List that will be used to concatenate all reviews with their respective topics into a dataframe

    concat_df = []
    resulting_df = [{}, {}]

    # Classify positive reviews into topics.
    df_positive = df[df[emotion_column] == "POSITIVE"]
    if df_positive.shape[0] > 0:
        positive_results = topic_modelling(df_positive, review_column, min_topic_size=min_topic_size, language=language)
        concat_df.append(df_positive)
        resulting_df[0]["POSITIVE"] = positive_results
        resulting_df[1]["POSITIVE"] = df_positive

    # Classify neutral reviews into topics
    df_neutral = df[df[emotion_column] == "NEUTRAL"]
    if df_neutral.shape[0] > 0:
        neutral_results = topic_modelling(df_neutral, review_column, min_topic_size=min_topic_size, language=language)
        concat_df.append(df_neutral)
        resulting_df[0]["NEUTRAL"] = neutral_results
        resulting_df[1]["NEUTRAL"] = df_neutral

    # Classify negative reviews into topics
    df_negative = df[df[emotion_column] == "NEGATIVE"]
    if df_negative.shape[0] > 0:
        negative_results = topic_modelling(df_negative, review_column, min_topic_size=min_topic_size, language=language)
        concat_df.append(df_negative)
        resulting_df[0]["NEGATIVE"] = negative_results
        resulting_df[1]["NEGATIVE"] = df_negative

    # Classify reviews that could be either negative or positive
    df_neg_pos = df[df[emotion_column] == "NEGATIVE-POSITIVE"]
    if df_neg_pos.shape[0] > 0:
        neg_pos_results = topic_modelling(df_neg_pos, review_column, min_topic_size=min_topic_size, language=language)
        concat_df.append(df_neg_pos)
        resulting_df[0]["NEGATIVE-POSITIVE"] = neg_pos_results
        resulting_df[1]["NEGATIVE-POSITIVE"] = df_neg_pos

    # Classify reviews that could be either negative or neutral
    df_neg_net = df[df[emotion_column] == "NEGATIVE-NEUTRAL"]
    if df_neg_net.shape[0] > 0:
        neg_net_results = topic_modelling(df_neg_net, review_column, min_topic_size=min_topic_size, language=language)
        concat_df.append(df_neg_net)
        resulting_df[0]["NEGATIVE-NEUTRAL"] = neg_net_results
        resulting_df[1]["NEGATIVE-NEUTRAL"] = df_neg_net

    # Classify reviews that could be either neutral or positive
    df_net_pos = df[df[emotion_column] == "NEUTRAL-POSITIVE"]
    if df_net_pos.shape[0] > 0:
        net_pos_results = topic_modelling(df_net_pos, review_column, min_topic_size=min_topic_size, language=language)
        concat_df.append(df_net_pos)
        resulting_df[0]["NEUTRAL-POSITIVE"] = net_pos_results
        resulting_df[1]["NEUTRAL-POSITIVE"] = df_net_pos
    
    # In case there is a review that has been asigned all three emotions. Examin its topics.
    df_all = df[df[emotion_column] == "NEGATIVE-NEUTRAL-POSITIVE"]
    if df_all.shape[0] > 0:
        all_results = topic_modelling(df_all, review_column, min_topic_size=min_topic_size, language=language)
        concat_df.append(df_all)
        resulting_df[0]["NEGATIVE-NEUTRAL-POSITIVE"] = all_results
        resulting_df[1]["NEGATIVE-NEUTRAL-POSITIVE"] = df_all

    # Concatenating all data frames

    df_complete = pd.concat(concat_df, ignore_index=True)

    resulting_df.append(df_complete)

    return resulting_df