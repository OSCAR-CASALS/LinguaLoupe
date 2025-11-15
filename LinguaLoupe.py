from src.run_pipeline import run_sentiment_pipeline
from src.generate_report import install_stopwords
import argparse
import os
import webbrowser

parser = argparse.ArgumentParser()
parser.add_argument("-ti", "--title", type=str, help="Title of the report,if not specified it will be the same as the file containig the collection of texts.",
                    default = "None", required=False)
parser.add_argument("-t", "--text_data", type=str, help="csv, json, jsonl, tsv or xlsx file with text data.", required=True)
parser.add_argument("-text_c", "--text_column", type=str, help="Column in TEXT_DATA which contains the texts to be analyzed",
                    required=True)
parser.add_argument("-ckt", "--Columns_to_Keep_Text", help="If there are any columns in TEXT_DATA you want to keep in Text.csv, specify them with this argument.",
                    required=False, action="append", default=[])
parser.add_argument("-gct", "--group_to_count_text", help="Columns in TEXT_DATA to count the number of different appereances, the result will be found in Summary.csv in the output directory.",
                    action="append", required=False, default=["emotion"])
parser.add_argument("-mt", "--mean_text", help="Columns in TEXT_DATA to compute the mean of in Summary.csv.",
                    action="append", required=False, default=[])
parser.add_argument("-st", "--sum_text", help="Columns in TEXT_DATA to sum in Summary.csv.",
                    action="append", required=False, default=[])
parser.add_argument("-umap_colour", "--umap_colour", help="Column in TEXT_DATA by which the umap shown in the report will be colored by, this parameter can be specified more than once in case you want to generate multiple UMAPs coloured by different values.",
                    default=["emotion"], action="append",
                    required=False)

parser.add_argument("-o", "--output_directory", type=str, help="Output directory, it will be the current working directory by default.", default=os.getcwd(),
                        required=False)
parser.add_argument("-csv_sep", "--csv_separation", type=str, help="In case a csv file is used as input, specify the separation between values, it will be ',' by default.",
                    default=",", required=False)
parser.add_argument("-min_rows_paralllelize", "--minimum_rows_paralllelize", type=int,
                    help="Minimum ammount of rows there must be for the program to parallelize computations via swifter, it will be 10,000 rows by default.", default=10000, required=False)
#parser.add_argument("-cancel_parallelisation", "--cancel_parallelisation", type=bool, help="Wether to avoid parallelisation with Swifter (True) or not (False) once a certain number of rows is found in TEXT_DATA, it will be False by default.",
#                    default=False, required=False)
parser.add_argument("-chunk_size", "--chunk_size", type=int, help="Chunk size in which the text will be divided when performing sentiment classification.",
                    default=512, required=False)
parser.add_argument("-min_topic_size", "--minimum_topic_size", type=int,
                    help="The minimum size of a topic. Increasing this value will lead to a lower number of clusters/topics and vice versa.",
                    default=10, required=False)
parser.add_argument("-lang", "--language", type=str, help="The main language used in your documents, it can be: 'english' (default), 'spanish', or 'portuguese'.", default="english", required=False)
parser.add_argument("-umap_metric", "--umap_metric", type=str, default="cosine", help="Metric to be used when computing distances for umap, will be cosine by default. You can check all avalaible metrics here: https://umap-learn.readthedocs.io/en/latest/parameters.html")
parser.add_argument("-umap_n_neighbours", "--umap_n_neighbours", type=int, default=15, help="Number of approximate nearest neighbors used to construct the UMAP, 15 by default.")
parser.add_argument("-umap_min_dist", "--umap_min_dist", type=float, default=0.1, help="Minimum distance apart that points are allowed to be in the umap, 0.1 by default.")
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

count_text_group = args.group_to_count_text
if ("emotion" in count_text_group) == False:
    count_text_group.append("emotion")
mean_text_cols = args.mean_text
sum_text_cols = args.sum_text
output_directory=os.path.join(args.output_directory, title)
csv_sep = args.csv_separation
min_rows_par = args.minimum_rows_paralllelize
#cancel_par = args.cancel_parallelisation
cancel_par = True
ch_size = args.chunk_size
m_topic_size = args.minimum_topic_size
lang = args.language
u_col = args.umap_colour
umap_metric_d = args.umap_metric
neighbours_umap = args.umap_n_neighbours
min_dist_umap = args.umap_min_dist

for c in u_col:
    if (c != "emotion") and (c not in cols_keep_text):
        print(f"{c} not in Columns_to_Keep_Text")
        exit()


run_sentiment_pipeline(text_data, title, text_col, cols_keep_text, count_text_group, mean_text_cols, sum_text_cols,
                       output_directory, csv_sep, min_rows_par, cancel_par, ch_size, m_topic_size, lang, umap_colour=u_col,
                       umap_metric=umap_metric_d, umap_neighbours=neighbours_umap, umap_minimum_distance=min_dist_umap)

absolute_path_to_html = os.path.abspath(output_directory)
webbrowser.open(f"file://{absolute_path_to_html}/report.html")