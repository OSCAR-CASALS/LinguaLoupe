'''
Script containing a function for running the whole LinguaLoupe pipeline.
'''

from src.reviews import process_reviews
from src.get_topics import review_topics
from src.collect_information import summerize_information
from src.generate_report import generate_report
import pandas as pd
import os

def run_sentiment_pipeline(text_data, title, text_col, cols_keep_text,
                           count_text_group, mean_text_cols, sum_text_cols,
                           output_directory, csv_sep, min_rows_par, cancel_par, ch_size, m_topic_size, lang, umap_colour = ["emotion"],
                           umap_metric="cosine", umap_neighbours = 15, umap_minimum_distance = 0.1):
    '''
    Run LinguaLoupe pipeline
    '''

    # Classify sentiments into negative, positive and neutral.
    print("Classifiying text into emotions...")
    reviews = process_reviews(text_data,
                            text_col, columns_to_keep=cols_keep_text, csv_sep=csv_sep,
                            min_rows_to_parallelize=min_rows_par, cancel_parallelisation=cancel_par, divide_in_chunks=ch_size,
                            convert_to_string=False, language=lang)
    
    # Perform topic modelling
    print("Dividing text into topics...")
    topics = review_topics(reviews, min_topic_size=m_topic_size, language=lang)

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

    TopicsDataFrame["topic"] = TopicsDataFrame["emotion"] + "_" +  TopicsDataFrame["topic"].astype(str)

    TopicsDataFrame.to_csv(os.path.join(output_directory,"Texts.csv"), sep=";", index=False)

    summary_data.to_csv(os.path.join(output_directory,"Summary.csv"), sep=";", index=False)

    # Most frequent topics
    for k in topics[0].keys():
        topics[0][k][1].to_csv(os.path.join(output_directory, k + ".csv"), sep=";", index=False)

    print("Generating html report...")
    
    return generate_report(title=title, review_dataframe=topics[-1], topic_models=topics[0],
                           path=output_directory, umap_summ_color=umap_colour, umap_met=umap_metric,
                           neighbours_umap=umap_neighbours, min_dist_umap=umap_minimum_distance)