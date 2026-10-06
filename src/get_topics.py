'''
Functions for topic analysis.
'''

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import pandas as pd
import warnings
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from umap import UMAP

def load_BERT(lang = "english",
              min_topic_size=10,
              n_neighbors=15,
              n_components=5,
              low_memory = True,
              embedding = "all-MiniLM-L6-v2",
              embedding_device = None):
    '''
    Creates a BERTtopic model using topic representation KeyBERTInspired.
    '''
    # Defining umap model for BERTopic
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine', low_memory=low_memory, init='random')

    # Creating BERTopic model
    representation_model = KeyBERTInspired()

    # Loading senetence transformer for embeddings
    embedding_model = SentenceTransformer(embedding, device=embedding_device)

    return BERTopic(
        language=lang,
        embedding_model=embedding_model,
        verbose=True,
        representation_model=representation_model,
        min_topic_size=min_topic_size,
        umap_model=umap_model
        )

def get_topics(model, df, reviews_columns):
    '''
    Uses a BERTtopic model to find topics in a dataframe with texts.
    '''
    docs = df[reviews_columns].to_list()
    return model.fit_transform(docs)

def topic_modelling(df, review_columns, min_topic_size=10, language="english", n_neighbors=15, n_components=5,
                    low_memory= True, embedding_device = None, embedding_model_name = "all-MiniLM-L6-v2"):
    '''
    Classifies reviews in different topics.
    '''
    # Loading model and dividing in topics
    topic_model = load_BERT(
        min_topic_size=min_topic_size,
        lang=language,
        n_neighbors=n_neighbors,
        n_components=n_components,
        low_memory=low_memory,
        embedding = embedding_model_name,
        embedding_device = embedding_device
        )
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
            return topic_modelling(
                df,
                review_columns,
                reduced_topic_size,
                language,
                n_neighbors,
                n_components,
                low_memory,
                embedding_device= embedding_device,
                embedding_model_name = embedding_model_name)
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

def review_topics(df, review_column = "text",emotion_column = "emotion", min_topic_size=10, min_topic_size_global = 10,
                  language="english", n_neighbors=15, n_components=5, low_memory= True, perform_sentiment_classification = True,
                  embedding_model_name = "all-MiniLM-L6-v2", embedding_device = None, n_neighbors_global = 15, n_components_global = 5,
                  Summarization_Model = None, min_summary_length = 14, max_summary_length = 40,
                  temperature_summ_model = 0.3):
    '''
    Divide positive, neutral and negative texts into topics.
    '''

    # List that will be used to concatenate all reviews with their respective topics into a dataframe

    concat_df = []
    resulting_df = [{}, {}]

    # Classifiying all texts into topics globally first
    Global_Topics = topic_modelling(df, review_column, min_topic_size=min_topic_size_global,
                                    language=language, n_neighbors=n_neighbors_global, n_components=n_components_global, low_memory=low_memory,
                                    embedding_model_name = embedding_model_name, embedding_device = embedding_device)

    df.rename(columns={'topic': 'global_topic', 'probability_topic': 'global_probability_topic'}, inplace=True)

    #Summarizing topics into sentences
    if Summarization_Model is not None:
        summarization = pipeline('text-generation', model = Summarization_Model, device_map="auto")
        sentence_summary = summarize_topics_into_sentences(
            bert_model=Global_Topics[0],
            summerization_model=summarization,
            max_output_length=max_summary_length,
            df_add_column=Global_Topics[1],
            temperature_summerization_model=temperature_summ_model
        )

    if perform_sentiment_classification == True:

        # Get diferent levels the gropu columns has
        levels = df[emotion_column].unique()

        # For each level, perform topic classification.
        for lev in levels:
            df_redux = df[df[emotion_column] == lev]
            results = topic_modelling(
                df_redux, review_column, min_topic_size=min_topic_size, language=language,
                n_neighbors=n_neighbors, n_components=n_components, low_memory=low_memory,
                embedding_model_name = embedding_model_name, embedding_device = embedding_device
            )

            if Summarization_Model is not None:
                semtiment_sentence_summary = summarize_topics_into_sentences(
                    bert_model=results[0],
                    summerization_model=summarization,
                    max_output_length=max_summary_length,
                    df_add_column=results[1],
                    temperature_summerization_model=temperature_summ_model
                )

            concat_df.append(df_redux)
            resulting_df[0][lev] = results
            resulting_df[1][lev] = df_redux

        # Concatenating all data frames

        df_complete = pd.concat(concat_df, ignore_index=True)
        
    else:
        df_complete = df

    resulting_df.append(df_complete)

    return [Global_Topics, resulting_df]

def summarize_topics_into_sentences(
        bert_model,
        summerization_model = None,
        max_output_length = 50,
        temperature_summerization_model = 0.3,
        df_add_column = None
        ):
    '''
    Create a sentence that summerized each topic with an LLM
    '''

    # Load summerization model
    if summerization_model is None:
        summerization_model = pipeline('text-generation', model = 'meta-llama/Llama-3.2-3B-Instruct', truncation = True, device_map="auto")

    # For each topic get most representative docs
    representative_docs = bert_model.get_representative_docs()

    #dictionary where results will be kept
    results = {}

    # Loop through topics and generate a sentence

    for topic, docs in representative_docs.items():

        # Put docs in a string
        documents = '\n- '.join(docs)

        # Put keywords of each topic in a string
        keywords_list = bert_model.get_topic(topic)
        keywords = ", ".join([word[0] for word in keywords_list])

        # Generate prompt
        prompt = f"""
You are an expert synthesizer.
Read the following texts that belong to the same topic.
Write a single sentence of maximum {max_output_length} words that captures the main idea or conclusion they have in common, without including secondary details.

Texts to summarize:
{documents}

You can use the following keywords as support:
{keywords}
"""

        sentence_topic = summerization_model(
            prompt,
            max_new_tokens = max_output_length*2,
            temperature = temperature_summerization_model,
            return_full_text=False,
            do_sample=True
            )[0]['generated_text']

        results[topic] = sentence_topic

        if df_add_column is not None:
            df_add_column.loc[df_add_column['Topic'] == topic,'LinguaLoupe_summary'] = sentence_topic

    return results
        


