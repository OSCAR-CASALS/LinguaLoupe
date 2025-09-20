'''
File containing the function that generates the html report.
'''

import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import umap
import umap.plot

def generate_report(title, review_dataframe, topic_models, lang="english", path = "", umap_summ_color = "emotion",
                    umap_met="cosine", neighbours_umap = 15, min_dist_umap = 0.1):
    '''
    Generate html report for an exploratory sentiment and topic analysis.
    '''

    # CSS
    css_template = '''
body {
  font-family: sans-serif;
  background-color: #F5F6FA;
  padding: 0;
}

.ContainsAll {
  display: flex;
  flex-direction: row;
  gap: 0;
  padding:0;
}

.ContainerWholeReport{
  padding:2%;
  width: 80%;
}

.navBar{
  padding: 2%;
  background-color: #2A4B7C;
  color: white;
  margin: 0;
  height: 100vh;
  position: fixed;
  z-index: 3;
  top:0;
  left:0;
  list-style-type: none;
}

.navBar li{
  margin-top: 6%;
  list-style-type: disc;
  margin-left: 10%;
}

.navBar h1{
  color: white;
}

.navBar a{
  text-decoration: none;
  color: white;
}

.navBar a:hover{
  color: #FF6B3D;
}

.navBar hr{
  background-color: white;
  border-style: solid;
  border-color: white;
}

.navBarBack{
  width: 18%;
  height: 0;
  background-color: #2A4B7C;
  color: white;
  margin: 0;
}

h1{
  text-align: center;
  width: 100%;
}

h1, h2, h3, h4{
  color: #2A4B7C;
}

a{
  color: #1E90FF;
}

a:hover{
  color: #FF6B3D;
}
footer{
    color: white;
    background-color: #2A4B7C;
    text-align: center;
    padding: 2%;
    margin: -3%;
    margin-top: 2%;
    width: 100%;
}
select, button{
    background-color: white;
    color: black;
    border-color: #2A4B7C;
    border-style: solid;
    border-width: medium;
    font-family: sans-serif;
    margin-bottom: 1%;
}

button:hover{
    background-color: #FF6B3D;
}

.dataTables_wrapper {
  font-family: sans-serif;
  color: black;
}
.dataTables_filter{
  font-family: sans-serif;
  color: black;
}
'''

    # HTML template script

    hide_button_function = '''{
    let el = document.getElementById(id_element);
    let b = document.getElementById(id_but);
    if (el.style.display == "none"){
        el.style.display = "block";
        b.innerHTML = "Hide Topics";
    } else{
        el.style.display = "none";
        b.innerHTML = "Show Topics";
    }
    console.log(id_element);
}
'''

    # HTML template

    html_template = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- Include jQuery (required by DataTables) -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>

    <!-- Include DataTables CSS (for styling the table) -->
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.css">

    <!-- Include DataTables JS -->
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.js"></script>
    <title>LinguaLoupe: {title}</title>
    <style>
    {css_template}
    </style>
    <script>
        function HideButton(id_element, id_but){hide_button_function}
    </script>
</head>
<body>
    <div class="ContainsAll">
    <ul class="navBar">
      <h1>LinguaLoupe</h1>
      <hr>
      <label style="font-weight: bold;">Go to:</label>
      <li><a href="#Summary">Summary</a></li>
      <li><a href="#SemtinmentPOSITIVE">Positive Texts Analysis</a></li>
      <li><a href="#SemtinmentNEUTRAL">Neutral Texts Analysis</a></li>
      <li><a href="#SemtinmentNEGATIVE">Negative Texts Analysis</a></li>
    </ul>
    <div class="navBarBack">
    </div>
    <div class="ContainerWholeReport">
    <main>
        <h1>{title}</h1>
'''

    # Colours of each emotion

    emotion_colours = {
        "POSITIVE": "#639754",
        "NEUTRAL": "#BDBABB",
        "NEGATIVE": "#D61F1F",
        "NEGATIVE-POSITIVE": "#FFD301",
        "NEGATIVE-NEUTRAL": "#8B0000",
        "NEUTRAL-POSITIVE": "#8B9A8B"
    }

    def generate_html_for_emotion(em):
        # Dataframe emotions
        em_df = review_dataframe[review_dataframe["emotion"] == em]
        # Trigrams
        cv = CountVectorizer(ngram_range=(3, 3), max_features=5000, min_df=5)
        bigrams = cv.fit_transform(em_df["text"])
        count_values = bigrams.toarray().sum(axis=0)
        ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv.vocabulary_.items()], reverse = True))
        ngram_freq.columns = ["Count", "Trigram"]
        barplot_trigrams = px.bar(ngram_freq.head(10),x="Count", y="Trigram",
                                  category_orders={"Trigram": ngram_freq["Trigram"].to_list()},
                                  title="Top 10 Trigrams")
        barplot_trigrams.update_layout(title={
            'text': '<b>Top 10 Trigrams</b>',
            'font': {'size': 24, 'family': 'Arial', 'color': 'black'},  # Size and font
            'x': 0.5,  # Centered title
        }, showlegend=False)
        barplot_trigrams.update_traces(marker_color=emotion_colours[em])
        bigram_plot_div = pio.to_html(barplot_trigrams,
                                      full_html=False, include_plotlyjs="cdn")


        if topic_models[em][1].shape[0] < 2:
            return f"<div><p>Could not find topics for {em}</p></div>"
        
        model = topic_models[em][0]
        # UMAP
        if topic_models[em][1].shape[0] < 2:
            umap_div = '<p style="color: red;">Not enougth topics to generate UMAP.</p>'
            hierarchical_div = '<p style="color: red;">Not enougth topics to perform hierarchical clustering.</p>'
        else:
            try:
                umap_em = model.visualize_topics()
                umap_div = pio.to_html(umap_em, full_html=False, include_plotlyjs="cdn")
            except Exception as e:
                umap_div = f'<p style="color: red;">Umap could not be generated: {e}.</p>'
            hierarchical_em = model.visualize_hierarchy()
            hierarchical_div = pio.to_html(hierarchical_em, full_html=False, include_plotlyjs="cdn")
        # Heatmap
        heatmap_em = model.visualize_heatmap()
        heatmap_div = pio.to_html(heatmap_em, full_html=False, include_plotlyjs="cdn")
        # Barchart of words
        barplot_em = model.visualize_barchart(top_n_topics=len(model.get_topics()))
        barplot_em.update_layout(title={
            'text': '<b>Topic Word Scores</b>',
            'font': {'size': 24, 'family': 'Arial', 'color': 'black'},  # Size and font
            'x': 0.5,  # Centered title
        })
        barplot_em.update_traces(marker_color=emotion_colours[em])
        barplot_div = pio.to_html(barplot_em, full_html=False, include_plotlyjs="cdn")
        # topic frequencies
        topics_freqs_dataframe = topic_models[em][1]
        topics_freqs_dataframe["Topic"] = topics_freqs_dataframe["Topic"].astype(str)
        topic_counts_barplot = px.bar(topics_freqs_dataframe, x="Topic", y="Count", title="Topic Counts")
        topic_counts_barplot.update_layout(title={
            'text': '<b>Topic Counts</b>',
            'font': {'size': 24, 'family': 'Arial', 'color': 'black'},  # Size and font
            'x': 0.5,  # Centered title
        }, showlegend=False)
        topic_counts_barplot.update_traces(marker_color=emotion_colours[em])
        barplot_topics_freq_div = pio.to_html(topic_counts_barplot, full_html=False, include_plotlyjs="cdn")
        # Showing reviews with each emotion
        column_data = [em_df[col].tolist() for col in em_df.columns]
        fig = go.Figure(data=[go.Table(
                    header=dict(values=list(em_df.columns),
                                fill_color='paleturquoise',
                                align='left'),
                    cells=dict(values=column_data,
                            fill_color='lavender',
                            align='left'))
                ])
        table_div = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
        # Making an interacteable table to show the diferent text included in the review
        table_emotion = em_df.to_html(index=False, classes="display", table_id=f"table_{em}")
        # Creating dropdown to filter table by topic
        dropdown_topic_menu = f'<label for="topics_filter_{em}">Topics:</label> <select name="topics_filter_{em}" id="topics_filter_{em}" onchange="Filter_Topic_{em}()"><option value="All">All</option>'
        for t in topics_freqs_dataframe["Topic"]:
            dropdown_topic_menu += f'<option value="{t}">{t}</option>'
        dropdown_topic_menu += "</select>"
        # javascript for the emotion
        style_sel_input = '''
{
              language: {
                    lengthMenu:
                        'Display <select style="background-color: white; color: black; border-color: #2A4B7C; border-style: solid; border-width: medium; font-family: sans-serif; margin-bottom: 1%;">' +
                        '<option value="5">5</option>' +
                        '<option value="10">10</option>' +
                        '<option value="15">15</option>' +
                        '<option value="20">20</option>' +
                        '<option value="25">25</option>' +
                        '<option value="30">30</option>' +
                        '<option value="35">35</option>' +
                        '<option value="40">40</option>' +
                        '<option value="45">45</option>' +
                        '<option value="50">50</option>' +
                        '</select> records'
                },
            pageLength : 5
            }
'''
        jav_script = "$(document).ready( function () {"
        jav_script += f"table{em} = $('#table_{em}')"
        jav_script += f".DataTable({style_sel_input});"
        jav_script += "});"
        # Function to filter table entries by topic
        fun_filter = "{"
        fun_filter += f'let f = document.getElementById("topics_filter_{em}").value;'
        fun_filter += "if (f == 'All'){"
        fun_filter += f"table{em}"
        fun_filter += '.column(-2).search("", true, false).draw();'
        fun_filter += '}else{'
        fun_filter += f"table{em}"
        fun_filter += '.column(-2).search(`^${f}` + "$", true, false).draw();}}'
        # resulting div
        res = f'''
        <div>
            <h3 id="Semtinment{em}">{em}</h3>
            <h4>Top 10 most frequent Trigrams</h4>
            {bigram_plot_div}
            <h4 id="TopicFreqs{em}">Topic Counts</h4>
            <p>Barplot displaying how many times a topic appears.</p>
            {barplot_topics_freq_div}
            <h4 id="Intertopic{em}">Intertopic distance map</h4>
            <p>A Umap visualization generated in a way very similar to <a href="https://github.com/cpsievert/LDAvis">LDAvis</a>.</p>
            {umap_div}
            <h4 id="Hclust{em}">Hierarchcial clustering</h4>
            <p>A graph displaying the potential hierarchy of topics.</p>
            {hierarchical_div}
            <h4 id="WordScore{em}">Topic Word Score</h4>
            <p>c-TF-IDF scores of each topic, you can visualize the most meaningfull words for each of the topics.</p>
            <button id="{em}_positive_but_id_topics_show" onclick="HideButton(this.value, this.id)" value="{em}_topics_barplots_tf">Show Topics</button>
            <div id="{em}_topics_barplots_tf" style="display: none;">
                {barplot_div}
            </div>
            <h4 id="Sim{em}">Similarity Matrix</h4>
            <p>A heatmap showing how similar the topics are between them.</p>
            {heatmap_div}
            <h4 id="table{em}">{em} texts</h4>
            <p>The texts that were considered to belong to: {em}</p>
            {dropdown_topic_menu}
            {table_emotion}
            <script>
            let table{em};
            {jav_script}
            function Filter_Topic_{em}(){fun_filter}
            </script>
        </div>
        '''
        return res


    # Histogram
    histogram_emotions = px.histogram(review_dataframe, x = "emotion", color="emotion", color_discrete_map=emotion_colours,
                                      labels={"emotion": "Emotion"}, title="Emotions")
    
    histogram_emotions.update_layout(showlegend=False, title={
        'text': '<b>Emotions</b>',
        'font': {'size': 24, 'family': 'Arial', 'color': 'black'},  # Size and font
        'x': 0.5,  # Centered title
    })
    div_histogram = pio.to_html(histogram_emotions, full_html=False, include_plotlyjs="cdn")

    # Boxplot review length
    df_emotion_length = pd.DataFrame()
    df_emotion_length["emotion"] = review_dataframe["emotion"]
    df_emotion_length["text_word_length"] = review_dataframe["text"].apply(lambda x: len(x.split()))
    
    boxplot_review_length = px.box(df_emotion_length, x="emotion", y="text_word_length", color="emotion", color_discrete_map=emotion_colours,
                                   labels={"emotion": "Emotion", "text_word_length": "Amount of words"}, title="Word Count")
    boxplot_review_length.update_layout(showlegend=False, title={
        'text': '<b>Word Count</b>',
        'font': {'size': 24, 'family': 'Arial', 'color': 'black'},  # Size and font
        'x': 0.5,  # Centered title
    })
    boxplot_review_length_div = pio.to_html(boxplot_review_length, full_html=False, include_plotlyjs="cdn")

    # Dimensionality reduction visualisation UMAP

    tfidf_vectorizer = TfidfVectorizer(min_df=5, stop_words=lang)
    tfidf_word_doc_matrix = tfidf_vectorizer.fit_transform(review_dataframe["text"])
    tfidf_umap = umap.UMAP(n_components=2,metric=umap_met, min_dist=min_dist_umap, n_neighbors=neighbours_umap)
    tfidf_embedding = tfidf_umap.fit_transform(tfidf_word_doc_matrix)
    
    umap_fig_em = px.scatter(tfidf_embedding[:,0], tfidf_embedding[:,1], color=review_dataframe[umap_summ_color], color_discrete_map=emotion_colours,
                             title="UMAP")
    umap_fig_em.update_layout(title = {
        'text': '<b>UMAP</b>',
        'font': {'size': 24, 'family': 'Arial', 'color': 'black'},  # Size and font
        'x': 0.5,  # Centered title
    })
    umap_fig_em_div = pio.to_html(umap_fig_em, full_html=False, include_plotlyjs="cdn")
    html_summary = f'''
    <div>
        <h2 id="Summary">Summary</h2>
        <h3 id="HistogramSentiment">Amount of texts per emotions</h3>
        <p>Histogram displaying the amount of text that belong to each emotion.</p>
        {div_histogram}
        <h3 id="WordCount">Amount of words per emotion</h3>
        <p>Boxplot showing the amount of words each text has, text have been divided by emotion.</p>
        {boxplot_review_length_div}
        <h3 id="UmapEmotions">Umap reviews colored by emotion</h3>
        <p>A UMAP to see how similar the text of each emotion are.</p>
        {umap_fig_em_div}
    </div>
'''

    emotion_sections = []

    # Topics plots
    for k in topic_models.keys():
        div_emotion = generate_html_for_emotion(k)
        emotion_sections.append(div_emotion)

    html_end = '''
</main>
    </div>
    </div>
    <footer>
        Report created using LinguaLoupe.
    </footer>
</body>
</html>
'''

    html_final = "\n".join([html_template, html_summary, "<h2>Sentiment Analysis</h2>","\n".join(emotion_sections), html_end])
    
    with open(os.path.join(path, "report.html"), "w+") as file:
        file.write(html_final)

    return html_final