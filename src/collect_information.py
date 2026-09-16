'''
File containing the function in charge of summarizing information.
'''

import pandas as pd

def summerize_information(review_dataframe, title, groups_to_count_reviews=[], columns_to_mean_review=[], columns_to_sum_reviews = []):
    '''
    Summerize text dataframe information into a dataframe of 1 row
    '''
    # Creating summary datframe
    df = pd.DataFrame(index=[0])

    # Adding title and description if present
    df["Title"] = title
    
    # Counting groups and adding them to the summary dataframe
    def add_dataframe(x, df_counts, dataframe_name):
        #print(x)
        df[f"ammount_of_{x}"] = df_counts[df_counts["Group"] == x]["Count"].values[0]
        
    for i in groups_to_count_reviews:
        counts_column = review_dataframe[i].value_counts().reset_index()
        counts_column.columns = ["Group", "Count"]
        counts_column["Group"].map(lambda y : add_dataframe(y, df_counts=counts_column, dataframe_name = "reviews"))

    # Averaging the columns specified to do so
        
    for i in columns_to_mean_review:
        df[f"average_{i}"] = review_dataframe[i].mean()

    # Columns to sum

    for i in columns_to_sum_reviews:
        df[f"count_{i}"] = review_dataframe[i].sum()

    # Review number is the number of rows in the review dataframe
    df["Number of texts"] = review_dataframe.shape[0]

    return df