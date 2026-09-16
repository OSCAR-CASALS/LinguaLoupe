'''
Script containing a function for running the whole LinguaLoupe pipeline.
'''

from src.reviews import process_reviews
from src.get_topics import review_topics
from src.collect_information import summerize_information
from src.generate_report import generate_report
import pandas as pd
import os
import re

def run_sentiment_pipeline(text_data, title, text_col, cols_keep_text,
                           count_text_group, mean_text_cols, sum_text_cols,
                           output_directory, csv_sep, ch_size, m_topic_size, lang, umap_colour = ["emotion"],
                           umap_metric="cosine", umap_neighbours = 15, umap_minimum_distance = 0.1, model_type = "social_media",
                           n_neighbours_BERTopic = 15, umap_n_components_BERTopic = 5, low_memory_BERTopic = True,
                           clean_html = True, include_tables = False, sentiment_classification = True, embedding_model_name = "all-MiniLM-L6-v2",
                           embedding_device = None, sentiment_device = -1,group_column = "emotion", m_topic_size_global = 10):
    '''
    Run LinguaLoupe pipeline
    '''

    # If group_column is not emotion, it means that the user has set up another column to devide the report in groups and therefore
    # it is not needed to perform sentiment classification.
    sent_class = sentiment_classification
    if group_column != "emotion":
        sent_class = False
    # Classify sentiments into negative, positive and neutral.
    print("Classifiying text into emotions...")
    reviews = process_reviews(text_data,
                            text_col, columns_to_keep=cols_keep_text, csv_sep=csv_sep,
                            divide_in_chunks=ch_size,
                            convert_to_string=False, language=lang, m_type=model_type, clean_html_text=clean_html,
                            perform_sentiment_classification=sent_class, sent_device=sentiment_device)

    if sentiment_classification:
        # Count ammount of text for each group.
        for gr in reviews[group_column].unique():
            print(f"Ammount of {gr} Texts: {reviews[reviews[group_column] == gr].shape[0]}")
    
    # Perform topic modelling
    print("Dividing text into topics...")
    topics_step = review_topics(
        reviews,
        min_topic_size=m_topic_size,
        language=lang,
        n_neighbors=n_neighbours_BERTopic,
        n_components=umap_n_components_BERTopic,
        low_memory=low_memory_BERTopic,
        perform_sentiment_classification=sentiment_classification,
        embedding_model_name = embedding_model_name,
        embedding_device = embedding_device,
        emotion_column=group_column,
        min_topic_size_global=m_topic_size_global
    )
    
    topics = topics_step[1]
    global_topic_model = topics_step[0][0]
    global_top_ten_topics = topics_step[0][1]
    
    # Summareize the information
    print("Creating csv files...")
    summary_data = summerize_information(review_dataframe=topics[-1],
                                    title=title, groups_to_count_reviews=count_text_group,
                                    columns_to_mean_review=mean_text_cols,
                                    columns_to_sum_reviews=sum_text_cols
                                    )

    # Save output

    if os.path.exists(output_directory) == False:
        os.mkdir(output_directory)

    TopicsDataFrame = topics[-1].copy()

    if sentiment_classification:
        TopicsDataFrame["topic"] = TopicsDataFrame[group_column] + "_" +  TopicsDataFrame["topic"].astype(str)

    TopicsDataFrame.to_csv(os.path.join(output_directory,"Texts.csv"), sep=";", index=False)

    summary_data.to_csv(os.path.join(output_directory,"Summary.csv"), sep=";", index=False)

    # Most frequent topics
    for k in topics[0].keys():
        # Sanitising file name so there are no issues when creating a file with the topic name
        file_name_topic_csv = re.sub(r'[<>:"/\\|?*]', '_', k)
        # Creating csv file
        topics[0][k][1].to_csv(os.path.join(output_directory, file_name_topic_csv + ".csv"), sep=";", index=False)

    # Global most frequent topics
    global_top_ten_topics.to_csv(os.path.join(output_directory, "Most_Frequent_Global_Topics.csv"), sep=";", index=False)

    print("Generating html report...")
    
    return generate_report(title=title, review_dataframe=topics[-1], topic_models=topics[0], Global_topic_Model=[global_topic_model, global_top_ten_topics],
                           path=output_directory, umap_summ_color=umap_colour, umap_met=umap_metric,
                           neighbours_umap=umap_neighbours, min_dist_umap=umap_minimum_distance, lang=lang,
                           include_table_emotions=include_tables, sentiment_classification=sentiment_classification, group_col=group_column)