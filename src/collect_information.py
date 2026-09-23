'''
File containing the function in charge of summarizing information.
'''

import pandas as pd

def summerize_information(review_dataframe, cols_to_group_by=[], columns_to_mean_review=[], columns_to_sum_reviews = []):
    '''
    Summerize text dataframe information.
    '''

    # if no lists, just return an empty dataframe.
    df = pd.DataFrame()
    if (len(cols_to_group_by) == 0) and (len(columns_to_mean_review) == 0) and (len(columns_to_sum_reviews) == 0):
        return df

    # Creating dictionary with columns to perform count, mean, and sum.
    agg_dictionary = {}

    copy_df = review_dataframe.copy()
    
    for col in columns_to_mean_review:
        new_c = f'{col} mean'
        copy_df[new_c] = review_dataframe[col]
        agg_dictionary[new_c] = 'mean'

    for col in columns_to_sum_reviews:
        new_c = f'{col} sum'
        copy_df[new_c] = review_dataframe[col]
        agg_dictionary[new_c] = 'sum'

    # Performing group by
    if len(cols_to_group_by) > 0:
        grouped = copy_df.groupby(cols_to_group_by)

        if len(agg_dictionary.keys()) > 0:
            df = grouped.agg(agg_dictionary)
            df['Number of texts'] = grouped.size()
            df = df.reset_index()
        else:
            df = grouped.size().reset_index(name='Number of texts')

        return df
    
    # If no group by is needed, just apply the transformations.
    
    if len(agg_dictionary.keys()) > 0:
        df = copy_df.agg(agg_dictionary).to_frame().T
    df['Number of texts'] = review_dataframe.shape[0]

    return df