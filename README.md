# LinguaLoupe

## Description

A data pipeline for sentiment analysis and topic classification, enabling to derive meaningful insights from large collections of textual data, such as:
reviews, online discussions, tweets, posts...

Specifically, it first divides all texts into topics and emotions allowing for an overview of the data and, afterwords, it performs topic classification on each sentiment or category defined by the user separately so a more in-depth analysis can be perfomred.

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
|-dt                    |--text_data                |TEXT_DATA                |csv, json, jsonl, tsv or xlsx file with text data.|
|-text_c                |--text_column              |TEXT_COLUMN              |Column in TEXT_DATA which contains the texts to be analyzed|
|-o                     |--output_directory         |OUTPUT_DIRECTORY         |Output directory, it will be the current working directory by default.|
|-lang                  |--language                 |LANGUAGE                 |The main language used in your documents, it can be: 'english' (default), or 'spanish'.|
|-mt                    |--model_type               |MODEL_NAME               |Whether to use a model for sentiment classification trained on social media data (use "social_media" option) or one fine-tuned for reviews (use "review" option), "social_media" is used by default.|

### Opional arguments

Additionally, you can set the following parameters so the report and csv files generated fit the data better.

|Abreviation                |Long argument               |Name                      |Description|
|---------------------------|----------------------------|--------------------------|-----------|
|-ti                        |--title                     |TITLE                     |Title of the report,if not specified it will be the same as the file containig the collection of texts.|
|-ckt                       |--Columns_to_Keep_Text      |COLUMNS_TO_KEEP_TEXT      |If there are any columns in TEXT_DATA you want to keep in _Text.csv_, specify them with this argument.   |
|-gbc                       |--group_by_column           |GROUP_BY_COLUMN           |Columns in TEXT_DATA by which to group by when summerizing the information, the results can be found in the file _Summary.csv_ inside the output directory. To group by more than one column define this argument multiple times (for example: -gbc Year -gbc Country).|
|-mean                        |--mean_text                 |MEAN_TEXT                 |Columns in TEXT_DATA to compute the mean of in _Summary.csv_. The mean will be computed separetly for each group obtained from GROUP_BY_COLUMN, if none has been specified this metric will be computed globally.|
|-sum                        |--sum_text                  |SUM_TEXT                  |Columns in TEXT_DATA to sum in _Summary.csv_. The addition will be computed separetly for each group obtained from GROUP_BY_COLUMN, if none has been specified this metric will be computed globally.|
|-umap_colour               |--umap_colour               |UMAP_COLOUR               |Column in COLUMNS_TO_KEEP_TEXT by which the umap shown in the report will be colored by, this parameter can be specified more than once in case you want to generate multiple UMAPs coloured by different values.|
|-col                       |--Category_Column              |CATEGORY_COLUMN              |Column by which to devide text in the report. By default it is a new column called 'emotion' created by the pipeline.|
|-csv_sep                   |--csv_separation            |CSV_SEPARATION            |In case a csv file is used as input, specify the separation between values, it will be "," by default.|
|-chunk_size                |--chunk_size                |CHUNK_SIZE                |Chunk size in which each text will be divided when performing sentiment classification. If not specified each text won't be divided in chunks when performing sentiment classification.|
|-min_topic_size            |--minimum_topic_size        |MINIMUM_TOPIC_SIZE        |The minimum size of a topic. Increasing this value will lead to a lower number of clusters/topics and vice versa. By default is 10.|
|-min_topic_size_global     |--minimum_topic_size_global |MINIMUM_TOPIC_SIZE_GLOBAL |The minimum size of a global topic. Increasing this value will lead to a lower number of clusters/topics and vice versa. By default it will have the same value as MINIMUM_TOPIC_SIZE|
|-umap_n_neighbors_BERTopic|--umap_n_neighbors_BERTopic|UMAP_N_NEIGHBORS_BERTOPIC|Number of approximate nearest neighbors used to construct the UMAP used in BERTopic, 15 by default.|
|-umap_n_components_BERTopic|--umap_n_components_BERTopic|UMAP_N_COMPONENTS_BERTOPIC|Number of components of the UMAP used in BERTopic, 5 by default.|
|-umap_metric               |--umap_metric               |UMAP_METRIC              |Metric to be used when computing distances for umap, will be cosine by default. You can check all avalaible metrics here: https://umap-learn.readthedocs.io/en/latest/parameters.html|
|-umap_n_neighbors         |--umap_n_neighbors         |UMAP_N_NEIGHBORS        |Number of approximate nearest neighbors used to construct the UMAP, 15 by default.|
|-umap_min_dist             |--umap_min_dist             |UMAP_MIN_DIST            |Minimum distance apart that points are allowed to be in the umap, 0.1 by default.|
|-e_model                   |--embedding_model           |EMBEDDING_MODEL          |Name or path of the model that will be used by BERTopic for embeddings through SentenceTransformers. If set to 'default', the program will use all-MiniLM-L6-v2 for english text and paraphrase-multilingual-MiniLM-L12-v2 for other languages.|
|-d                  |--device           |DEVICE        |Device to be used for sentiment classification and embedding texts in topic classification. By default it will check if there are gpu avalaible (autodetect), if not, it will use cpu. If you want to specify a specific device you can either set it to cpu (it will use cpu regardless of if there are gpu avalaible) or cuda (utilizes an NVIDIA graphics card).|

### Optional flag arguments

There are certain true and false parameters that affect how the pipeline works and what is shown in the report. These arguments just need to be specified in the command line to change the default behaviour of LinguaLoupe to better suit the analysis being performed.

|Abreviation                |Long argument               |Name                      |Description|
|---------------------------|----------------------------|--------------------------|-----------|
|-high_memory_BERTopic       |--umap_high_memory_BERTopic|UMAP_HIGH_MEMORY_BERTOPIC |Add this flag when datasets may not consume a lot of memory or you want to not use low_memory UMAPs for BERTopic. Using millions of documents can lead to memory issues therefore low memory UMAPs are used by default when using BERTopic to alleviate some of them.|
|-not_clean_html            |--not_clean_html            |NOT_CLEAN_HTML            |By default html characters are removed from the text column prior to sentiment and topic classification. Set this flag to not remove them.|
|-s_tables                  |--show_tables               |SHOW_TABLES               |By default the html report does not show the texts belonging to each topic, sentiment or group. Set this flag to show them.|
|-r_classification          |--remove_classification     |REMOVE_CLASSIFICATION     |If you are just interested in a global topic classification, without wanting to divide the texts in any way for a more in-depth analysis, set this flag so only the topic classification corresponding to all texts is performed. The report will only show the 'Summary' section.|


### Example

Below there is an example on how to use the pipeline for sentiment and topic analysis on a dataset composed of tweets:

```
python LinguaLoupe.py -ti GlobalWarmingTwitter -dt twitter_sentiment_data.csv -text_c tweets -min_topic_size 100 -o results -lang english -mt social_media
```

If you want to ignore sentiment classification and use another column from your dataset instead for the more in-depth analysis, the argument CATEGORY_COLUMN (-col) must be set with the name of the column you want to use.

Below is an example of how LinguaLoupe can be used to find different topics across multiple news, each belonging to a different class.

```
python LinguaLoupe.py -ti News_topics -dt news.csv -text_c Description -min_topic_size 100 -o results -lang english -col Class
```

### Output

The pipeline will always generate the following 4 files, being _report.html_ the most important one, these are:

- **_report.html_**: An html report showing through different plots and tables the results of the sentyment and topic analysis, as well as the topic analysis per sentiment.

- **_Summary.csv_**: A csv file displaying the ammount of positive, neutral and negative texts; as well as the group counts, means, and sums the user has specified through arguments: GROUP_TO_COUNT_TEXT, MEAN_TEXT and SUM_TEXT respectively.

- **_Texts.csv_**: A csv file with all the texts considered in the analysis, it contains the following columns:
    + _text_: The texts that have been sentiment and topic analyzed.
    + _CATEGORY_COLUMN_: This is the column by which texts have been divided for a more in depth topic analysis. When sentiment classification is performed this will be the column _emotion_ and it will be suplemented by another column called _emotion\_score_ displaying the likelyhood of the text belonging to the sentiment it has been classified as.
    + Any column specified by the user in COLUMNS_TO_KEEP_TEXT.
    + _global\_topic_: The topic a specific text belongs to according to the topic classification performed on all texts.
    + _global_probability_topic__: The probability of a text to belong to it's assigned _global\_topic_. 
    + _topic_: The topic the text has been included inside the category it belongs to (either _emotion_ or categories defined by the user with the argument CATEGORY_COLUMN).
    + _probability\_topic_: The probability of a text to belong to it's assigned _topic_.

- **_Most\_Frequent\_Global\_Topics.csv_**: A csv file containing information about the topics detected in the topic classification performed on all texts. Specifically it displays:
    + The ammount of times each topic appears.
    + The main words of each topic.
    + The c-TF-IDF score of each main word.

Additionally, if the flag REMOVE_CLASSIFICATION is not set, a csv for each category in CATEGORY_COLUMN will be created, each will contain:
- The ammount of times each topic appears.
- The main word of each topic.
- The c-TF-IDF score of each main word.

## Structure

This repository is divided as following:

|File or Directory name|Description|
|----------------------|-----------|
|LinguaLoupe.py        |The python script that must be executed to run the pipeline.|
|src                   |Directory containing all functions the pipeline uses.|
|requirements.txt      |The tools needed to run the program.|
|LICENSE.txt           |License of the program.|


## Tools used for sentiment and Topic classification.

- For sentyment classification, depending on the language and the model type the following pretrained models are used: 
    + **social_media**:
        + English: cardiffnlp/twitter-roberta-base-sentiment_
        + Spanish: _pysentimiento_.
    + **review**:
        + English: _siebert/sentiment-roberta-large-english_
        + Spanish: _nlptown/bert-base-multilingual-uncased-sentiment_
- For topic classification BERTopic was used.

## Current Version

LinguaLoupe is currently on version v.0.4.0-alpha.

## Whats new compared to the previous version?

- The _Summary.csv_ generated in the output has been overhauled, now the gbc option has been added so users can group by any category they want and count the number of appereances of each group in the dataset, as well as compute the mean or sum of any numerical column they want in the dataset.

- Fixed a major bug where there could not be a column already named _emotion_ in TEXT_DATA.

- As a consecuence from updating _Summary.csv_, it's generation has become faster.

## Planned updates

The following updates are planned for end of 2027:

- Desktop app that will allow to use the pipeline outside the Command Line.
- Improve documentation and create new one for the Desktop app.
- Add more customization to the report.
- Create a portable binary and executable (.exe) of LinguaLoupe so it can be used without the need of python.
- Create examples of how this tool can be used with public datasets and how it can be used as a key piece for workflows focused on text analysis.
- Add more options to customize the  global topic classification separately from the one performed for each category. 

## Warnings

- As of now, if a column called emotion exists in TEXT_DATA aside from the one generated by LinguaLoupe, it's name will be changed to _emotion\_original_ in order to not raise issues with the column _emotion_ generated by the pipeline when performing sentiment classification; a more propper fix will come on later releases.

## Citations

- Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. In Findings of the Association for Computational Linguistics: EMNLP 2020 (pp. 1644–1650). Association for Computational Linguistics.

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.

- Pérez, J. M., Rajngewerc, M., Giudici, J. C., Furman, D. A., Luque, F., Alemany, L. A., & Martínez, M. V. (2023). pysentimiento: A Python Toolkit for Opinion Mining and Social NLP tasks. arXiv [Cs.CL]. Retrieved from http://arxiv.org/abs/2106.09462

- NLP Town. (2023). bert-base-multilingual-uncased-sentiment (Revision edd66ab). doi:10.57967/hf/1515


