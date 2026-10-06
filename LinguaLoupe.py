from src.run_pipeline import run_sentiment_pipeline
from src.generate_report import install_stopwords
from src.emotion_colors import colors_report
import argparse
import os
import torch
import json

parser = argparse.ArgumentParser()
parser.add_argument("-ti", "--title", type=str, help="Title of the report,if not specified it will be the same as the file containig the collection of texts.",
                    default = "None", required=False)
parser.add_argument("-dt", "--text_data", type=str, help="csv, json, jsonl, tsv or xlsx file with text data.", required=True)
parser.add_argument("-text_c", "--text_column", type=str, help="Column in TEXT_DATA which contains the texts to be analyzed",
                    required=True)

parser.add_argument("-mn", "--model_name", type=str, help="The hugging face model you want to use for sentiment classification. By default LinguaLoupe uses cardiffnlp/twitter-roberta-base-sentiment.", default="cardiffnlp/twitter-roberta-base-sentiment")
parser.add_argument("-labels", "--labels", type=str, default=None, help='If you want to map the labels outputed by the MODEL_NAME selected to something else, provide the path of a JSON file where the keys are the labels outputed by the hugging face model and the values the names you want to assign each of them in the report and the CSV files.')
parser.add_argument("-ckt", "--Columns_to_Keep_Text", help="If there are any columns in TEXT_DATA you want to keep in Text.csv, specify them with this argument.",
                    required=False, action="append", default=[])
parser.add_argument("-gbc", "--group_by_column", help="Columns in TEXT_DATA by which to group by when summerizing the information, the results can be found in the file Summary.csv inside the output directory.",
                    action="append", required=False, default=["emotion"])
parser.add_argument("-mean", "--mean_text", help="Columns in TEXT_DATA to compute the mean of in Summary.csv.",
                    action="append", required=False, default=[])
parser.add_argument("-sum", "--sum_text", help="Columns in TEXT_DATA to sum in Summary.csv.",
                    action="append", required=False, default=[])
parser.add_argument("-umap_colour", "--umap_colour", help="Column in TEXT_DATA by which the umap shown in the report will be colored by, this parameter can be specified more than once in case you want to generate multiple UMAPs coloured by different values.",
                    default=["emotion"], action="append",
                    required=False)
parser.add_argument("-col", "--Category_Column", required=False, default="emotion", type=str, help="Column by which to divide text in the report. By default it is a new column called 'emotion' created by the pipeline.")

parser.add_argument("-o", "--output_directory", type=str, help="Output directory, it will be the current working directory by default.", default=os.getcwd(),
                        required=False)
parser.add_argument("-csv_sep", "--csv_separation", type=str, help="In case a csv file is used as input, specify the separation between values, it will be ',' by default.",
                    default=",", required=False)
parser.add_argument("-chunk_size", "--chunk_size", type=int, help="Chunk size in which the text will be divided when performing sentiment classification. If not specified each text won't be divided in chunks when performing sentiment classification.",
                    default=None, required=False)
parser.add_argument("-min_topic_size", "--minimum_topic_size", type=int,
                    help="The minimum size of a topic. Increasing this value will lead to a lower number of clusters/topics and vice versa. By default is 10.",
                    default=10, required=False)
parser.add_argument("-min_topic_size_global", "--minimum_topic_size_global", type=int,
                    help="The minimum size of a global topic. Increasing this value will lead to a lower number of clusters/topics and vice versa. By default it will have the same value as MINIMUM_TOPIC_SIZE",
                    default=None, required=False)

parser.add_argument("-umap_n_neighbors_BERTopic", "--umap_n_neighbors_BERTopic", type=int, default=15, help="Number of approximate nearest neighbors used to construct the UMAP used in BERTopic, 15 by default.")
parser.add_argument("-umap_n_neighbors_BERTopic_global", "--umap_n_neighbors_BERTopic_global", type=int, help="Number of approximate nearest neighbors used to construct the UMAP required by BERTopic for the global topic classification. By default it will be the same value as UMAP_N_NEIGHBORS_BERTOPIC.", default = None)
parser.add_argument("-umap_n_components_BERTopic", "--umap_n_components_BERTopic", type=int, default=5, help="Number of components of the UMAP used in BERTopic, 5 by default.")
parser.add_argument("-umap_n_components_BERTopic_global", "--umap_n_components_BERTopic_global", type=int, help="Number of components of the UMAP required by BERTopic for the global topic classification. By default it will be the same value as UMAP_N_COMPONENTS_BERTOPIC.", default = None)

parser.add_argument("-high_memory_BERTopic", "--umap_high_memory_BERTopic", action="store_false", help="Add this flag when datasets may not consume a lot of memory or you want to not use low_memory UMAPs for BERTopic. Using millions of documents can lead to memory issues therefore low memory UMAPs are used by default when using BERTopic to alleviate some of them.")
# n_neighbors=15, n_components=5, low_memory= True

parser.add_argument("-lang", "--language", type=str, help="The main language used in your documents, it can be any of the ones accepted by BERTopic.", default="english", required=False)
parser.add_argument("-umap_metric", "--umap_metric", type=str, default="cosine", help="Metric to be used when computing distances for umap, will be cosine by default. You can check all avalaible metrics here: https://umap-learn.readthedocs.io/en/latest/parameters.html")
parser.add_argument("-umap_n_neighbors", "--umap_n_neighbors", type=int, default=15, help="Number of approximate nearest neighbors used to construct the UMAP, 15 by default.")
parser.add_argument("-umap_min_dist", "--umap_min_dist", type=float, default=0.1, help="Minimum distance apart that points are allowed to be in the umap, 0.1 by default.")

parser.add_argument("-not_clean_html", "--not_clean_html", action="store_false", help="By default html characters are removed from the text column prior to sentiment and topic classification. Set this flag to not remove them.")
parser.add_argument("-s_tables", "--show_tables", action="store_true", help="By default the html report does not show the texts belonging to each topic, sentiment or group. Set this flag to show them.")
parser.add_argument("-r_classification", "--remove_classification", action="store_false", help="If you are just interested in a global topic classification, without wanting to divide the texts in any way for a more in-depth analysis, set this flag so only the topic classification corresponding to all texts is performed. The report will only show the 'Summary' section.")
parser.add_argument("-e_model", "--embedding_model", default="default", help="Name or path of the model that will be used by BERTopic for embeddings through SentenceTransformers. If set to 'default', the program will use all-MiniLM-L6-v2 for english text and paraphrase-multilingual-MiniLM-L12-v2 for other languages.")
parser.add_argument("-d", "--device", default="autodetect", help="Device to be used for sentiment classification and embedding texts in topic classification. By default it will check if there are gpu avalaible (autodetect), if not, it will use cpu. If you want to specify a specific device you can either set it to cpu (it will use cpu regardless of if there are gpu avalaible) or cuda (utilizes an NVIDIA graphics card)")

parser.add_argument("-colors", "--report_colors", type=str, default=None, help="Path to a Json file specifiying colours to be used for each category in the final html report. It must have the categories as keys and the colours as values.")
parser.add_argument("-summarize", "--summarization_model", type=str, default=None, help="A text-generation hugging face model to summerize the most representative texts of all topics into single sentences that will appear in Global_Topics.csv. If not set no summerization of topics will be performed.")
parser.add_argument("-temperature", "--summarization_model_temperature", type=float, default=0.3, help="Temperature of SUMMARIZATION_MODEL.")

args = parser.parse_args()

# Check if nltk stopwords are installed
install_stopwords()

# Defining input arguments
text_data = args.text_data
title = args.title
if title == "None":
    title = text_data.split(".")[0]
text_col = args.text_column
cols_keep_text = args.Columns_to_Keep_Text

r_classification = args.remove_classification
count_text_group = args.group_by_column
gr_col = args.Category_Column

if ("emotion" not in count_text_group) and ((r_classification == True) or (gr_col == "emotion")):
    count_text_group.append("emotion")
elif ((r_classification == False) or (gr_col != "emotion")):
    count_text_group.remove("emotion")

if (gr_col != "emotion") and (gr_col not in cols_keep_text):
    cols_keep_text.append(gr_col)


mean_text_cols = args.mean_text
sum_text_cols = args.sum_text

for c in count_text_group:
    if c not in cols_keep_text:
        cols_keep_text.append(c)

for c in mean_text_cols:
    if c not in cols_keep_text:
        cols_keep_text.append(c)

for c in sum_text_cols:
    if c not in cols_keep_text:
        cols_keep_text.append(c)

output_directory=os.path.join(args.output_directory, title)
csv_sep = args.csv_separation
cancel_par = True
ch_size = args.chunk_size

if ch_size is None:
    print("Text won't be divided in chunks when performing sentiment classification.")
else:
    print(f"Text will be divided in chunks of {ch_size} when performing sentiment classification.")

m_topic_size = args.minimum_topic_size
m_topic_size_global = args.minimum_topic_size_global

if m_topic_size_global is None:
    m_topic_size_global = m_topic_size

print(f"Minimum global topic size set to: {m_topic_size_global}")
print(f"Minimum topic size set to: {m_topic_size}")

lang = args.language
u_col = args.umap_colour

if ((r_classification == False) or (gr_col != "emotion")) and ("emotion" in u_col):
    u_col.remove("emotion")

if (r_classification == True) and (gr_col not in u_col):
    u_col.append(gr_col)

umap_metric_d = args.umap_metric
neighbours_umap = args.umap_n_neighbors
min_dist_umap = args.umap_min_dist
m_type = args.model_name

n_neighbours_BERTopic = args.umap_n_neighbors_BERTopic
n_components_BERTopic = args.umap_n_components_BERTopic

n_neighbours_BERTopic_global = args.umap_n_neighbors_BERTopic_global
n_components_BERTopic_global = args.umap_n_components_BERTopic_global

if n_neighbours_BERTopic_global is None:
    n_neighbours_BERTopic_global = n_neighbours_BERTopic

if n_components_BERTopic_global is None:
    n_components_BERTopic_global = n_components_BERTopic

print(f"Number of approximate nearest neighbors for UMAP used in global topic classification set to: {n_neighbours_BERTopic_global}")
print(f"Number of approximate nearest neighbors for UMAP used in topic classification set to: {n_neighbours_BERTopic}")

print(f"Number of components for UMAP used in global topic classification set to: {n_components_BERTopic_global}")
print(f"Number of components for UMAP used in topic classification set to: {n_components_BERTopic}")

l_memory = args.umap_high_memory_BERTopic
c_html = args.not_clean_html
show_tables = args.show_tables

emodel = args.embedding_model
edevice = args.device
sent_dev = args.device
summ_model = args.summarization_model
temp_model_summary = args.summarization_model_temperature

print(f'Summarization model set to: {summ_model}. Temperature = {temp_model_summary}')

print(f"Show tables set to: {show_tables}, Clean html set to: {c_html}")

for c in u_col:
    if (c != "emotion") and (c not in cols_keep_text):
        print(f"{c} not in Columns_to_Keep_Text")
        exit()

# Check if EMBEDDING_MODEL has been set to default, if that's the case two between 
# the two default options based on whether the language is english or not.

if emodel == "default":
    emodel = "paraphrase-multilingual-MiniLM-L12-v2"
    if lang == "english":
        emodel = "all-MiniLM-L6-v2"

print(f"Embedding model that will be used in topic classification: {emodel}")

if edevice == "autodetect":
    edevice = None
    if torch.cuda.is_available():
        sent_dev = "cuda"
    else:
        sent_dev = "cpu"

print(f"Device that will be used: {sent_dev}")


# Load JSON file with emotion labels if one has been defined

labels_dictionary = None

if m_type == "cardiffnlp/twitter-roberta-base-sentiment":
    labels_dictionary = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE"
    }


labels_dict_path = args.labels

if labels_dict_path is not None:
    with open(labels_dict_path, 'r') as f:
        labels_dictionary = json.load(f)

#Load json file with report colors.

colors_dictionary = {
    "POSITIVE": "#639754",
    "NEUTRAL": "#BDBABB",
    "NEGATIVE": "#D61F1F"
}

report_cols = args.report_colors

if report_cols is not None:
    with open(report_cols, 'r') as e:
        colors_dictionary = json.load(e)

colors_report.set_colors(colors_dictionary)


run_sentiment_pipeline(text_data = text_data,
                       title = title,
                       text_col = text_col,
                       cols_keep_text = cols_keep_text,
                       count_text_group = count_text_group,
                       mean_text_cols = mean_text_cols,
                       sum_text_cols = sum_text_cols,
                       output_directory = output_directory,
                       csv_sep = csv_sep,
                       ch_size = ch_size,
                       m_topic_size = m_topic_size,
                       lang = lang,
                       umap_colour=u_col,
                       umap_metric=umap_metric_d,
                       umap_neighbours=neighbours_umap,
                       umap_minimum_distance=min_dist_umap,
                       model_type=m_type,
                       n_neighbours_BERTopic=n_neighbours_BERTopic,
                       umap_n_components_BERTopic=n_components_BERTopic,
                       low_memory_BERTopic=l_memory,
                       clean_html=c_html, include_tables=show_tables,
                       sentiment_classification=r_classification,
                       embedding_model_name=emodel,
                       embedding_device=edevice,
                       group_column=gr_col,
                       sentiment_device=sent_dev,
                       m_topic_size_global = m_topic_size_global,
                       n_neighbours_BERTopic_global=n_neighbours_BERTopic_global,
                       umap_n_components_BERTopic_global=n_components_BERTopic_global,
                       emotion_labels=labels_dictionary,
                       summarization_model = summ_model,
                       temperature_summarization_model = temp_model_summary)

#absolute_path_to_html = os.path.abspath(output_directory)
#webbrowser.open(f"file://{absolute_path_to_html}/report.html")