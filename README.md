# LinguaLoupe

## Description

A data pipeline for sentiment analysis and topic modelling, enabling to derive meaningful insights from large collections of textual data, such as:
reviews, online discussions, tweets, posts...

The results are shown mainly in an html report complemented by a few csv files.

## Installation

### Conda

To install the dependencies required to run the program you can create a _Conda_ environment from the yaml file provided with the follwing command:

```
conda env create -f environment.yml
```

### Pip

Alternatively, you can install the dependencies required to run the program with the
file requirements.txt as following:

```
pip install -r requirements.txt
```

## Usage

To run the pipeline, it is just needed to specify the following arguments and run LinguaLoupe.py:

|Abreviation            |Long argument              |Name                     |Description|
|-----------------------|---------------------------|-------------------------|---------|
|-t                     |--text_data                |TEXT_DATA                |csv, json, jsonl, tsv or xlsx file with text data.|
|-text_c                |--text_column              |TEXT_COLUMN              |Column in TEXT_DATA which contains the texts to be analyzed|
|-o                     |--output_directory         |OUTPUT_DIRECTORY         |Output directory, it will be the current working directory by default.|
|-mo                    |--model_type               |MODEL_TYPE               |Whether to use a model for sentiment classification trained on social media data (use "social_media" option) or a review model (use "review" option), "social_media" is used by default.|

Additionally, you can set the following parameters so the report and csv files generated fit the data better.

|Abreviation                |Long argument               |Name                      |Description|
|---------------------------|----------------------------|--------------------------|-----------|
|-ti                        |--title                     |TITLE                     |Title of the report,if not specified it will be the same as the file containig the collection of texts.|
|-ckt                       |--Columns_to_Keep_Text      |COLUMNS_TO_KEEP_TEXT      |If there are any columns in TEXT_DATA you want to keep in _Text.csv_, specify them with this argument.   |
|-gct                       |--group_to_count_text       |GROUP_TO_COUNT_TEXT       |Columns in TEXT_DATA to count the number of different appereances, the result will be found in _Summary.csv_.|
|-mt                        |--mean_text                 |MEAN_TEXT                 |Columns in TEXT_DATA to compute the mean of in _Summary.csv_.|
|-st                        |--sum_text                  |SUM_TEXT                  |Columns in TEXT_DATA to sum in _Summary.csv_.|
|-umap_colour               |--umap_colour               |UMAP_COLOUR               |Column in COLUMNS_TO_KEEP_TEXT by which the umap shown in the report will be colored by, this parameter can be specified more than once in case you want to generate multiple UMAPs coloured by different values.|
|-csv_sep                   |--csv_separation            |CSV_SEPARATION            |In case a csv file is used as input, specify the separation between values, it will be "," by default.|
|-min_rows_paralllelize     |--minimum_rows_paralllelize |MINIMUM_ROWS_PARALLLELIZE |Minimum ammount of rows there must be for the program to parallelize computations via swifter, it will be 10,000 rows by default.|
|-chunk_size                |--chunk_size                |CHUNK_SIZE                |Chunk size in which each text will be divided when performing sentiment classification.|
|-min_topic_size            |--minimum_topic_size        |MINIMUM_TOPIC_SIZE        |The minimum size of a topic. Increasing this value will lead to a lower number of clusters/topics and vice versa.|
|-lang                      |--language                  |LANGUAGE                  |The main language used in your documents, it can be: 'english' (default), or 'spanish'.|
|-umap_n_neighbours_BERTopic|--umap_n_neighbours_BERTopic|UMAP_N_NEIGHBOURS_BERTOPIC|Number of approximate nearest neighbors used to construct the UMAP used in BERTopic, 15 by default.|
|-umap_n_components_BERTopic|--umap_n_components_BERTopic|UMAP_N_COMPONENTS_BERTOPIC|Number of components of the UMAP used in BERTopic, 5 by default.|
|-low_memory_BERTopic       |--umap_low_memory_BERTopic  |UMAP_LOW_MEMORY_BERTOPIC  |Using millions of documents can lead to memory issues and setting this value to True might alleviate some of the issues.|
|-umap_metric               |--umap_metric               |UMAP_METRIC              |Metric to be used when computing distances for umap, will be cosine by default. You can check all avalaible metrics here: https://umap-learn.readthedocs.io/en/latest/parameters.html|
|-umap_n_neighbours         |--umap_n_neighbours         |UMAP_N_NEIGHBOURS        |Number of approximate nearest neighbors used to construct the UMAP, 15 by default.|
|-umap_min_dist             |--umap_min_dist             |UMAP_MIN_DIST            |Minimum distance apart that points are allowed to be in the umap, 0.1 by default.|
|-clean_html                |--clean_html                |CLEAN_HTML               |Whether to remove html characters from text_column or not, True by default.|

Below there is an example on how to use the pipeline:

```
python LinguaLoupe.py -ti GlobalWarmingTwitter -t example/twitter_sentiment_data.csv -text_c message -ckt tweetid -min_topic_size 100 -o results -lang english -mo social_media
```

**Warning: paralllelization options are desactivated right now and will be implemented in the future.**

### Output

The pipeline generates 6 files, being _report.html_ the most important one, these are:

- **_report.html_**: An html report showing through different plots and tables the results of the sentyment and topic analysis, as well as the topic analysis per sentiment.

- **_Summary.csv_**: A csv file displaying the ammount of positive, neutral and negative texts; as well as the group counts, means, and sums the user has specified through arguments: GROUP_TO_COUNT_TEXT, MEAN_TEXT and SUM_TEXT respectively.

- **_Texts.csv_**: A csv file with all the texts considered in the analysis, it contains the following columns:
    + _text_: The texts that have been sentiment and topic analyzed.
    + _emotion_: The emotion they have been classified as in the analysis.
    + _emotion\_score_: The likelyhood of the text belonging to the emotion it has been classified as.
    + _topic_: The topic the text has been included in.
    + _probability\_topic_: The probability of the text belonging to the topic it has been assigned to.
    + Any column specified by the user in COLUMNS_TO_KEEP_TEXT.

- **_POSITIVE.csv_**: A csv file containing topic information for texts classified as positive, specifically it has:
    + The ammount of times each topic appears.
    + The main word of each topic.
    + The c-TF-IDF score of each main word.

- **_NEUTRAL.csv_**: A csv file containing topic information for texts classified as neutral, specifically it has:
    + The ammount of times each topic appears.
    + The main word of each topic.
    + The c-TF-IDF score of each main word.

- **_NEGATIVE.csv_**: A csv file containing topic information for texts classified as negative, specifically it has:
    + The ammount of times each topic appears.
    + The main word of each topic.
    + The c-TF-IDF score of each main word.

- **_Most\_Frequent\_Global\_Topics.csv_**: A csv file containing topic information for texts classified as negative, specifically it has:
    + The ammount of times each topic appears.
    + The main words of each topic.
    + The c-TF-IDF score of each main word.

## Structure

This repository is divided as following:

|File or Directory name|Description|
|----------------------|-----------|
|LinguaLoupe.py        |The python script that must be executed to run the pipeline.|
|src                   |Directory containing all functions the pipeline uses.|
|requirements.txt      |The tools needed to run the program.|
|example               |Example output|
|LICENSE.txt           |License of the program.|


## Tools used for sentiment and Topic classification.

- For sentyment classification, depending on the language and the model type the following pretrained models are used: 
    + **social_media**:
        + English: cardiffnlp/twitter-roberta-base-sentiment_
        + Spanish: _pysentimiento_.
    + **review**:
        + English: _siebert/sentiment-roberta-large-english_
        + Spanish: _nlptown/bert-base-multilingual-uncased-sentiment_
- For Topic modelling BERTopic was used.

## Planned updates

The following updates are planned for end of 2027:

- Desktop app that will allow to use the pipeline outside the Command Line.
- Support for other languages aside from english.

## Citations

- Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. In Findings of the Association for Computational Linguistics: EMNLP 2020 (pp. 1644–1650). Association for Computational Linguistics.

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.

- Pérez, J. M., Rajngewerc, M., Giudici, J. C., Furman, D. A., Luque, F., Alemany, L. A., & Martínez, M. V. (2023). pysentimiento: A Python Toolkit for Opinion Mining and Social NLP tasks. arXiv [Cs.CL]. Retrieved from http://arxiv.org/abs/2106.09462

- NLP Town. (2023). bert-base-multilingual-uncased-sentiment (Revision edd66ab). doi:10.57967/hf/1515


