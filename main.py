# for file-handling
import os
import warnings

# for downloading arxiv-taxonomy
import requests
from bs4 import BeautifulSoup

# for search-patterns
import re

# data-handling
import numpy as np
import pandas as pd

# for visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# for tokenization
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import TreebankWordTokenizer

# for vectorisation
from sklearn.feature_extraction.text import TfidfVectorizer

# for feature-reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# for clustering / topic-generation
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# for gensim topic modeling
from gensim import corpora
from gensim.models import Phrases
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

# for top2vec topic modeling
from top2vec import Top2Vec

# for visualization
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()


# basis functions
def read_dataset():
    """
    Loads downloaded arXiv dataset (.json) into pandas dataframe

    :return: arXiv dataset in pandas dataframe
    """
    if os.path.exists('arxiv-metadata-oai-snapshot.json'):
        # Reading of the provided data into a Pandas dataframe
        df = pd.read_json('arxiv-metadata-oai-snapshot.json', lines=True)
        # Reading of the rebuild data into a Pandas dataframe (for fast processing)
        return df
    else:
        raise FileNotFoundError


def read_taxonomy():
    """
    Code retrieved from https://www.kaggle.com/code/steubk/arxiv-taxonomy-e-top-influential-papers

    :return:
    """
    website_url = requests.get('https://arxiv.org/category_taxonomy').text
    soup = BeautifulSoup(website_url, 'xml')

    root = soup.find('div', {'id': 'category_taxonomy_list'})

    tags = root.find_all(["h2", "h3", "h4", "p"], recursive=True)

    level_1_name = ""
    level_2_code = ""
    level_2_name = ""

    level_1_names = []
    level_2_codes = []
    level_2_names = []
    level_3_codes = []
    level_3_names = []
    level_3_notes = []

    for t in tags:
        if t.name == "h2":
            level_1_name = t.text
            level_2_code = t.text
            level_2_name = t.text
        elif t.name == "h3":
            raw = t.text
            level_2_code = re.sub(r"(.*)\((.*)\)", r"\2", raw)
            level_2_name = re.sub(r"(.*)\((.*)\)", r"\1", raw)
        elif t.name == "h4":
            raw = t.text
            level_3_code = re.sub(r"(.*) \((.*)\)", r"\1", raw)
            level_3_name = re.sub(r"(.*) \((.*)\)", r"\2", raw)
        elif t.name == "p":
            notes = t.text
            level_1_names.append(level_1_name)
            level_2_names.append(level_2_name)
            level_2_codes.append(level_2_code)
            level_3_names.append(level_3_name)
            level_3_codes.append(level_3_code)
            level_3_notes.append(notes)

    df = pd.DataFrame({
        'group_name': level_1_names,
        'archive_name': level_2_names,
        'archive_id': level_2_codes,
        'category_name': level_3_names,
        'category_id': level_3_codes,
        'category_description': level_3_notes

    })

    return df


def save_dataset(df):
    """
    Saves the edited dataframe

    :param df: The dataframe to be saved
    :return: None
    """
    df.to_json('dataset_rebuild.json', orient='records', lines=True, date_format='iso')

    return


def save_taxonomy(taxonomy):
    """
    Saves the used taxonomy

    :param taxonomy: The loaded taxonomy as dataframe
    :return: None
    """
    taxonomy.to_json('arxiv_taxonomy.json', orient='records', lines=True)

    return


# FEATURE ENGINEERING FUNCTIONS
def preprocess_update_date(df):
    """
    Transforms the existing text date field into datetime format.
    (Only consideration of year and month to simplify evaluation).

    :param df: The dataframe to be edited
    :return: The edited dataframe
    """

    df['update_date'] = df['update_date'].apply(lambda x: x[:7])
    df['year'] = df['update_date'].apply(lambda x: x[:4])
    df['year'] = pd.to_numeric(df['year'])
    df['update_date'] = pd.to_datetime(df['update_date'], format='%Y-%m')

    return df


def preprocess_categories(df, taxonomy):
    """
    Adds the columns of group_names and category_ids to the dataframe according to the arxiv taxonomy.
    Column content is the number of mentions / assignment in the existing column 'categories'.
    In addition, the number of assigned categories is counted and assigned to 'Category Counts'.

    :param df: The dataframe to be edited
    :param taxonomy: Valid arxiv taxonomy
    :return: The edited dataframe
    """
    for group in taxonomy['group_name'].unique():
        categories = taxonomy.loc[taxonomy[taxonomy['group_name'] == group].index, 'category_id'].tolist()
        for category in categories:
            df[category] = df['categories'].apply(lambda x: 1 if category in x.split(' ') else 0)
        df[group] = df[categories].sum(axis=1)

    df['Category Counts'] = df['categories'].apply(lambda x: len(x.split(' ')))

    return df


def preprocess_word_counts(df):
    """
    The number of words in the 'title' and 'abstract' columns are counted and
    stored in the 'n_words_title' and 'n_words_abstract' columns.

    :param df: The dataframe to be edited
    :return: The edited dataframe
    """
    df['n_words_title'] = df['title'].apply(lambda x: len(x.split(' ')))
    df['n_words_abstract'] = df['abstract'].apply(lambda x: len(x.split(' ')))

    return df


# ANALYSE-FUNKTIONEN
# .............................. HIER NOCHMAL SCHAUEN! ..............................
def analysis_basic(df):
    """
    In the basic analysis, the available columns and the percentage of NaN values are printed.

    :param df: The dataframe to be edited
    :return: None
    """
    print('.............................. Base Analysis ..............................\n')
    print('Dataframe-Shape: Dataset has {} rows / entries and {} attributes.\n'
          .format(df.shape[0], df.shape[1]))

    # Analyzing Null-Values for every column
    s_df_nan = pd.Series(((df.isna().sum() / df.shape[0]) * 100).round(2),
                         index=df.columns,
                         name='Percentage of NaN-values per column')
    s_df_nan.rename_axis('Columns in Dataset', inplace=True)

    print('NaN-Value-Percentage for every column:\n{}\n'.format(s_df_nan))

    return


# FILTER FUNCTIONS
def filter_update_date(df, begin='2021-01-01', end='2022-12-31'):
    """
    The dataframe is filtered based on the date (begin-date - end-date).

    :param df: The dataframe to be edited
    :param begin: Enclosing start date ('YYYY-mm-dd') of the records to be considered within the data frame.
    :param end: Enclosing end date ('YYYY-mm-dd') of the records to be considered within the data frame.
    :return: The edited dataframe
    """

    df_entries = df.groupby('year').count()['id']

    # PLOT-DIAGRAMM ZU GESAMTENTWICKLUNG
    fig, ax = plt.subplots(figsize=(17/2.54, 5/2.54), dpi=300)
    ax.plot(df_entries.index, df_entries, linestyle='-', marker='o')

    plotline = ax.lines[0]
    xdata = plotline.get_xdata()
    ydata = plotline.get_ydata()
    color = plotline.get_color()

    left = int(begin[:4])
    right = int(end[:4])
    line_height_left = df_entries[df_entries.index == left]
    line_height_right = df_entries[df_entries.index == right]

    ax.set_ylim(ymin=ydata.min()-0.1*ydata.min())
    ax.vlines(left, ydata.min(), line_height_left, color=color, ls=':')
    ax.vlines(right, ydata.min(), line_height_right, color=color, ls=':')
    ax.fill_between(xdata, ax.get_ylim()[0], ydata, where=(xdata <= left), interpolate=True, facecolor='red', alpha=0.4)
    ax.fill_between(xdata, ax.get_ylim()[0], ydata, where=(xdata >= right), facecolor='red', alpha=0.4)

    ax.set_title('Annual development of entries in the dataset \n'
                 '(red area = filtered out)')
    ax.set_xlabel('Years in dataset')
    ax.set_ylabel('Entries (accumulated \nover the years)')
    ax.set_xticks(df_entries.index[::2])
    ax.set_ylim(ymin=ydata.min()-0.1*ydata.min())
    plt.savefig('figure_01_entries_date_filter', bbox_inches='tight')
    plt.show()

    df.drop(df[df['update_date'] < begin].index, inplace=True)
    df.drop(df[df['update_date'] >= end].index, inplace=True)

    return df


def analysis_word_counts(df):
    """
    KDE-Plot-Analysis according to the word counts of textual description in dataset

    :param df: The dataframe to be analyzed
    :return: None
    """
    # Create KDE-Plot of number of words in title column
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    sns.kdeplot(x=df['n_words_title'], bw_adjust=2, cut=0, ax=ax)
    kdeline = ax.lines[0]
    xdata = kdeline.get_xdata()
    ydata = kdeline.get_ydata()
    color = kdeline.get_color()
    left_quant = df['n_words_title'].quantile(.25)
    median = df['n_words_title'].quantile(.5)
    right_quant = df['n_words_title'].quantile(.75)
    height = np.interp(median, xdata, ydata)
    ax.vlines(median, 0, height, color=color, ls=':')
    ax.fill_between(xdata, 0, ydata, facecolor=color, alpha=0.4)
    ax.fill_between(xdata, 0, ydata, where=(left_quant <= xdata) & (xdata <= right_quant),
                    interpolate=True, facecolor=color, alpha=0.4)
    ax.set(title='KDE-Plot of number of words in title column \n'
                 '(with highlighting 0.25%, 0.5% and 0.75% quantile)',
           xlabel='Number of words')
    plt.savefig('figure_02_kdeplot_n_words_title')
    plt.show()

    # Create KDE-Plot of number of words in abstract column
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    sns.kdeplot(x=df['n_words_abstract'], bw_adjust=2, cut=0, ax=ax)
    kdeline = ax.lines[0]
    xdata = kdeline.get_xdata()
    ydata = kdeline.get_ydata()
    color = kdeline.get_color()
    left_quant = df['n_words_abstract'].quantile(.25)
    median = df['n_words_abstract'].quantile(.5)
    right_quant = df['n_words_abstract'].quantile(.75)
    height = np.interp(median, xdata, ydata)
    ax.vlines(median, 0, height, color=color, ls=':')
    ax.fill_between(xdata, 0, ydata, facecolor=color, alpha=0.4)
    ax.fill_between(xdata, 0, ydata, where=(left_quant <= xdata) & (xdata <= right_quant),
                    interpolate=True, facecolor=color, alpha=0.4)
    ax.set(title='KDE-Plot of number of words in abstract column \n'
                 '(with highlighting 0.25%, 0.5% and 0.75% quantile)',
           xlabel='Number of words')
    ax.set_xticks(range(0, 801, 100))
    plt.savefig('figure_03_kdeplot_n_words_abstract')
    plt.show()

    return


def analysis_taxonomy(taxonomy):
    """
    Analysis of the taxonomy in terms of groups and the number of subordinated categories.
    Graphic is created and saved.
    :param taxonomy: Valid arxiv taxonomy
    :return: None
    """
    colors_bar = sns.color_palette('bright', n_colors=len(taxonomy['group_name'].unique().tolist())+1)
    del colors_bar[3]

    n_unique_groups = len(taxonomy['group_name'].unique().tolist())
    n_unique_categories = len(taxonomy['category_id'].unique())

    # BALKENDIAGRAMM ZUR TAXONOMY
    fig, ax = plt.subplots(dpi=300)
    sns.countplot(data=taxonomy, y='group_name', palette=colors_bar,
                  order=taxonomy['group_name'].value_counts().index, ax=ax)
    ax.set(title='Allocation of all categories (n={}) to the groups (n={}):'.format(n_unique_categories,
                                                                                    n_unique_groups),
           xlabel='Number of categories per group',
           ylabel='Group')
    for container in ax.containers:
        ax.bar_label(container)
    plt.savefig('figure_04_taxonomy', bbox_inches='tight')
    plt.show()

    return


def analysis_categories(df, taxonomy):
    """
    Graphical analysis of the columns in the dataframe with respect to
    groups and categories according to arxiv taxonomy.

    The following analyses are performed:
    (1) Group mentions over the course of time
    (2) Share of the top 3 groups and the rest (cumulative)

    :param df: The dataframe to be analyzed
    :param taxonomy: Valid arxiv taxonomy
    :return: None
    """
    # CREATE NEW DATAFRAME GROUPED BY UPDATE-DATE
    df_group_analysis = df.groupby('update_date').sum()[taxonomy['group_name'].unique().tolist()]

    # SORT COLUMNS FROM HIGHEST COUNT (FIRST COLUMN) TO LOWEST COUNT (RIGHT COLUMN)
    df_group_analysis = df_group_analysis.loc[:, df_group_analysis.sum().sort_values(ascending=False).index]
    df_group_analysis_sma = df_group_analysis.rolling(3).mean().shift(-1)

    # DEFINING THE COLORS
    colors_plot = sns.color_palette('dark', n_colors=2)
    colors_pie_bar = sns.color_palette('bright', n_colors=len(taxonomy['group_name'].unique().tolist())+1)

    # PLOT-DIAGRAMM ZU GESAMTENTWICKLUNG
    fig, ax = plt.subplots(dpi=300)
    ax.plot(df_group_analysis.index, df_group_analysis.sum(axis=1), color=colors_plot[0], label='monthly mentions')
    ax.plot(df_group_analysis_sma.index[1:-1], df_group_analysis_sma.sum(axis=1)[1:-1],
            color=colors_plot[1], label='SMA 3 months')
    plt.legend()
    plt.savefig('figure_05_entries', bbox_inches='tight')
    plt.show()

    # TORTENDIAGRAMM ZU DEN GRUPPEN
    fig, ax = plt.subplots(dpi=300)

    ax.pie((df_group_analysis.iloc[:, 0].sum(),
            df_group_analysis.iloc[:, 1].sum(),
            df_group_analysis.iloc[:, 2].sum(),
            df_group_analysis.iloc[:, 3:].sum().sum()),
           labels=(df_group_analysis.columns[0],
                   df_group_analysis.columns[1],
                   df_group_analysis.columns[2],
                   'Other Groups'),
           explode=(0, 0, 0, 0.15),
           autopct='%.1f%%',
           colors=colors_pie_bar,
           shadow=True)
    ax.set_title('Classification of the groups in the dataset\n'
                 '({} group mentions for a total of {} records)'.format(df_group_analysis.sum().sum(), df.shape[0]))
    # ax.legend(loc=2, bbox_to_anchor=(-1, 1))
    plt.savefig('figure_06_pie_chart_groups', bbox_inches='tight')
    plt.show()

    # BAR-CHART TO OTHER GROUPS
    sum_other_groups = df_group_analysis.iloc[:, 3:].sum().sum()
    bottom = 0

    fig, ax = plt.subplots(figsize=(3.93, 8.27), dpi=300)

    for idx in range(3, df_group_analysis.shape[1]):
        value = df_group_analysis.iloc[:, idx].sum() / sum_other_groups
        # ax.bar('Other Groups', value, bottom=bottom)
        ax.bar('Other Groups', value, color=colors_pie_bar[idx+1],
               label=df_group_analysis.columns[idx] + ' (' + str(round(value * 100, 2)) + '%)', bottom=bottom)
        bottom += value

    # ax.tick_params(axis='both', which='both', length=0)
    ax.axis('off')
    ax.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Group names', loc=7, bbox_to_anchor=(2.5, .5))
    plt.savefig('figure_07_bar_chart_other_groups', bbox_inches='tight')
    plt.show()

    return


# TRANSFORMATION FUNCTIONS
def abstract_tokenzier_nltk(text, stop_words):
    """
    Tokenization of the input text with NLTK

    :param text: Text input to be tokenized
    :param stop_words: Pre-defined stop words
    :return: Tokenized text
    """
    tokenzier = TreebankWordTokenizer()
    wnl = WordNetLemmatizer()

    text = text.replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)

    tokens = tokenzier.tokenize(text)

    tokens = [word.lower() for word in tokens]

    # remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]

    # Lemmatisieren Sie jede Token, indem Sie die Wortarten taggen
    tagged_tokens = nltk.pos_tag(tokens)

    # Bestimmen Sie die Wortart jedes Tokens und wenden Sie die entsprechende Lemmatizer-Funktion an
    lemma_tokens = []
    for token, tag in tagged_tokens:
        if tag.startswith('NN'):
            lemma_token = wnl.lemmatize(token, wordnet.NOUN)
        else:
            lemma_token = token

        lemma_tokens.append(lemma_token)

    tokenized_text = " ".join([i for i in lemma_tokens])

    return tokenized_text


def tfidf_vectorizer(df):
    """
    Vectorization with TF-IDF algorithm from sklearn

    :param df:
    :return: Sparse-matrix X and feature_names
    """
    vectorizer = TfidfVectorizer(max_df=0.9,
                                 ngram_range=(1, 3),
                                 max_features=4096)
    X = vectorizer.fit_transform(df['processed_abstract'])

    feature_names = vectorizer.get_feature_names_out()

    return X, feature_names


def pca_feature_reduction(X):
    """
    Reduction of the features of the TF-IDF sparse matrix for faster computation of KMeans

    :param X: Sparse-matrix X
    :return: Reduced matrix X_reduced
    """
    pca = PCA(n_components=0.9, random_state=42)
    X_reduced = pca.fit_transform(X.toarray())

    return X_reduced


def kmeans_validation(X, min_topics, max_topics):
    """
    Determination of the optimal number of clusters / topics with KElbowVisualizer by Yellowbrick.
    Display of calculations incl. optimal k

    :param X: Reduced matrix from PCA-Feature-Reduction
    :param min_topics: Minimum number of clusters / topics
    :param max_topics: Maximum  number of clusters / topics
    :return: None
    """
    kmeans_model = KMeans()
    visualizer = KElbowVisualizer(kmeans_model, k=(min_topics, max_topics), timings=False)
    visualizer.fit(X)
    visualizer.show()

    return


def kmeans_modeling(df, X, n_topics):
    """
    Execution of KMeans based on the reduced matrix (TF-IDF - PCA) and
    optimal k for the number of clusters / topics determined via KElbowVisualizer.

    Assigned cluster / topic is written in dataframe column 'KMeans Topic'.

    :param df: The dataframe to be edited
    :param X: Reduced matrix from PCA-Feature-Reduction
    :param n_topics: Number of clusters / topics
    :return: The edited dataframe
    """
    kmeans_model = KMeans(n_clusters=n_topics, random_state=42)
    y_pred = kmeans_model.fit_predict(X)

    df['KMeans Topic'] = y_pred

    return df


def get_kmeans_top_words(cluster_num, X, feature_names, n_top_words=20):
    """
    Determines the top 20 most important words based on all entries in
    the dataframe with identical cluster / topic according to KMeans.

    :param cluster_num: Assigned Cluster / Topic by KMeans
    :param X: TF-IDF Matrix
    :param feature_names: Feature names from TF-IDF
    :param n_top_words: Number of top words
    :return: List of top 20 words
    """
    cluster_matrix = X[df_dataset['KMeans Topic'] == cluster_num]
    cluster_scores = cluster_matrix.sum(axis=0).A1
    top_indices = cluster_scores.argsort()[::-1][:n_top_words]
    top_words = [feature_names[i] for i in top_indices]

    return top_words


def lda_preprocessing(df):
    """
    Preparation of dictionary and corpus for gensim lda modeling

    :param df:
    :return: Gensim dictionary and corpus (BagOfWords vectorization)
    """
    abstracts = df['processed_abstract_tokenized'].tolist()
    # Add bigrams and trigrams to docs (only ones that appear 20 times or more)
    ngrams = Phrases(abstracts)
    for i in range(len(abstracts)):
        for token in ngrams[abstracts[i]]:
            if '_' in token:
                # Token is a bigram, add to document.
                abstracts[i].append(token)

    dictionary = corpora.Dictionary(abstracts)
    dictionary.filter_extremes(no_above=0.9, keep_n=4096)
    corpus = [dictionary.doc2bow(abstract) for abstract in abstracts]

    return dictionary, corpus


def lda_validation(df, corpus, dictionary, min_alpha, max_alpha, min_eta, max_eta, min_topics, max_topics):
    """
    Determination of optimal parameters (alpha, eta, number of topics) for
    LDA topic modeling according to coherence (c_v and c_nmpi).
    For each iteration from min_topics to max_topics a graph is created to compare the different parameters.

    :param df:
    :param corpus: Gensim corpus
    :param dictionary: Gensim dictionary
    :param min_alpha: Minimum alpha value
    :param max_alpha: Maximum alpha value
    :param min_eta: Minimum eta value
    :param max_eta: Maximum eta value
    :param min_topics: Minimum number of topics
    :param max_topics: Maximum number of topics
    :return: Best parameters as dictionary ('Number of Topics', 'alpha', 'eta') for both c_v and c_nmpi coherence model
    """
    abstracts = df['processed_abstract_tokenized'].tolist()

    max_c_v_config = {}
    max_c_nmpi_config = {}

    max_value_c_v = -100
    max_value_c_nmpi = -100

    alpha_values = np.arange(min_alpha, max_alpha + .01, step=.05)
    eta_values = np.arange(min_eta, max_eta + .01, step=.05)

    for alpha in alpha_values:
        for eta in eta_values:
            n_topics = []
            coherence_cv = []
            coherence_cnmpi = []

            print(f'Iterations for alpha = {alpha} and eta = {eta}:')

            for num_topics in range(min_topics, max_topics+1):
                lda_model = LdaModel(corpus=corpus,
                                     id2word=dictionary,
                                     num_topics=num_topics,
                                     alpha=alpha,
                                     eta=eta,
                                     iterations=500,
                                     passes=7)

                coherence_model_c_v = CoherenceModel(model=lda_model,
                                                     texts=abstracts,
                                                     dictionary=dictionary,
                                                     coherence='c_v'
                                                     )
                coherence_c_v = coherence_model_c_v.get_coherence()

                coherence_model_c_nmpi = CoherenceModel(model=lda_model,
                                                        texts=abstracts,
                                                        dictionary=dictionary,
                                                        coherence='c_npmi'
                                                        )
                coherence_c_npmi = coherence_model_c_nmpi.get_coherence()

                n_topics.append(num_topics)
                coherence_cv.append(coherence_c_v)
                coherence_cnmpi.append(coherence_c_npmi)

                if coherence_c_v > max_value_c_v:
                    max_value_c_v = coherence_c_v
                    max_c_v_config['Number of Topics'] = num_topics
                    max_c_v_config['alpha'] = alpha
                    max_c_v_config['eta'] = eta

                if coherence_c_npmi > max_value_c_nmpi:
                    max_value_c_nmpi = coherence_c_npmi
                    max_c_nmpi_config['Number of Topics'] = num_topics
                    max_c_nmpi_config['alpha'] = alpha
                    max_c_nmpi_config['eta'] = eta

                print(f"Number of clusters: {num_topics}\n"
                      f"Coherence c_v: {coherence_c_v}\n"
                      f"Coherence c_npmi: {coherence_c_npmi}")

            filename_c_v = 'coherence_c_v_alpha_'+str(int(alpha*100))+'_eta_'+str(int(eta*100))
            filename_c_nmpi = 'coherence_c_nmpi_alpha_'+str(int(alpha*100))+'_eta_'+str(int(eta*100))

            fig, ax = plt.subplots()
            ax.plot(n_topics, coherence_cv)
            plt.savefig(filename_c_v)
            plt.show()

            fig, ax = plt.subplots()
            ax.plot(n_topics, coherence_cnmpi)
            plt.savefig(filename_c_nmpi)
            plt.show()

    return max_c_v_config, max_c_nmpi_config


def lda_modeling(df, corpus, dictionary, num_topics, alpha, eta):
    """
    Implementation of gensim lda topic modeling.
    Assigning the topic number and keywords to the columns ...

    :param df:
    :param corpus: Gensim corpus
    :param dictionary: Gensim dictionary
    :param num_topics: Number of topics
    :param alpha: alpha value
    :param eta: eta value
    :return:
    """
    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         alpha=alpha,
                         eta=eta,
                         iterations=750,
                         passes=10)

    df['LDA Topic'] = [max(lda_model[corpus[i]],
                           key=lambda x: x[1])[0] for i in range(len(corpus))]

    keywords = lda_model.show_topics(num_topics=num_topics, num_words=20, formatted=False)
    keywords_dict = {i: [t[0] for t in keywords[i][1]] for i in range(num_topics)}

    df['LDA Keywords'] = [keywords_dict[topic] for topic in df['LDA Topic']]

    p = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(p, 'lda_model.html')

    return


def top2vec_modeling(df, n_topics):
    """
    Implementation of Top2Vec Topic Modeling.
    Saves the model for re-analysis.

    :param df:
    :param n_topics: Number of topics to which the model is to be reduced
    :return:
    """
    abstracts = df['processed_abstract'].tolist()
    idx = df['id'].tolist()
    top2vec_model = Top2Vec(documents=abstracts, ngram_vocab=True, document_ids=idx, speed='fast-learn')
    top2vec_model.save('top2vec_topic_model')

    topic_nums, topic_score, topic_words, words_scores = \
        top2vec_model.get_documents_topics(top2vec_model.document_ids)
    df['Top2Vec (ALL) Topic'] = topic_nums
    df['Top2Vec (ALL) Keywords'] = topic_words.tolist()
    df['Top2Vec (ALL) Keywords'] = df['Top2Vec (ALL) Keywords'].apply(lambda words: words[:20])

    top2vec_model.hierarchical_topic_reduction(n_topics)

    topic_nums, topic_score, topic_words, words_scores = \
        top2vec_model.get_documents_topics(top2vec_model.document_ids, reduced=True)
    df['Top2Vec (REDUCED) Topic'] = topic_nums
    df['Top2Vec (REDUCED) Keywords'] = topic_words.tolist()
    df['Top2Vec (REDUCED) Keywords'] = df['Top2Vec (REDUCED) Keywords'].apply(lambda words: words[:20])

    return top2vec_model


def tsne_visualization(X, topics, n_topics_model):
    """
    Visualization of vectorization by means of t-SNE algorithm

    :param X: Matrix from TF-IDF or BoW
    :param topics:
    :param n_topics_model: Number of topics according to model to be visualized
    :return: None
    """
    tsne = TSNE(verbose=1, perplexity=100, random_state=42)
    X_embedded = tsne.fit_transform(X.toarray())

    sns.set(rc={'figure.figsize': (15, 15)})
    palette = sns.hls_palette(n_topics_model, l=.4, s=.9)
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=topics, legend='full', palette=palette)
    plt.title('t-SNE with KMeans-Topics')
    plt.savefig('figure_08_t-SNE')
    plt.show()

    return


def reassign_topics(df, model):
    """
    Reassigns topic numbers from 1 (highest count) to X (=number of topics) descending

    :param df: Dataframe where topics are stored
    :param model: Name of the model
    :return: Edited dataframe
    """
    column_name = model + ' Topic'

    df_topic = df[column_name].value_counts().to_frame()
    new_idx = [x+1 for x in range(0, df_topic.shape[0])]
    df_topic['New Index'] = new_idx
    df[column_name] = df[column_name].apply(lambda x: df_topic.loc[x, 'New Index'])

    return df


def analysis_topics(df, kmeans_topics, lda_topics, top2vec_topics, n_top_words):
    """
    Analyzing the topics determined by KMeans, LDA and Top2Vec.
    - Distribution of topics per model
    - Average change per topic and model
    - Keyword matches
    - Wordclouds of most important topics (manually determined)

    :param df: The dataframe to be analyzed
    :param kmeans_topics: Number of KMeans-Topics
    :param lda_topics: Number of LDA-Topics
    :param top2vec_topics: Number of Top2Vec-Topics
    :param n_top_words: Number of keywords
    :return: None
    """

    kmeans_topic_total_perc = df['KMeans Topic'].value_counts(normalize=True)
    lda_topic_total_perc = df['LDA Topic'].value_counts(normalize=True)
    top2vec_topic_total_perc = df['Top2Vec (REDUCED) Topic'].value_counts(normalize=True)

    df_topics_total_perc = pd.DataFrame({'KMeans': kmeans_topic_total_perc,
                                         'LDA': lda_topic_total_perc,
                                         'Top2Vec': top2vec_topic_total_perc},
                                        index=range(1, max(kmeans_topics, lda_topics, top2vec_topics)+1))

    color_heatmap = sns.diverging_palette(22, 220, as_cmap=True)

    sns.heatmap(data=df_topics_total_perc.transpose(),
                cmap=color_heatmap,
                cbar=False,
                square=True,
                annot=True,
                fmt='.1%')
    plt.title('Distribution of topics for the 3 models')
    plt.xlabel('Topic Number')
    plt.savefig('figure_09_heatmap_topic_distribution')
    plt.show()

    # TIME SERIES
    kmeans_topic_trend_perc = \
        df.groupby('update_date')['KMeans Topic'].value_counts().unstack().pct_change(periods=1).mean()
    lda_topic_trend_perc = \
        df.groupby('update_date')['LDA Topic'].value_counts().unstack().pct_change(periods=1).mean()
    top2vec_topic_trend_perc = \
        df.groupby('update_date')['Top2Vec (REDUCED) Topic'].value_counts().unstack().pct_change(periods=1).mean()

    df_topics_trend_perc = pd.DataFrame({'KMeans': kmeans_topic_trend_perc,
                                         'LDA': lda_topic_trend_perc,
                                         'Top2Vec': top2vec_topic_trend_perc},
                                        index=range(1, max(kmeans_topics, lda_topics, top2vec_topics)+1))

    sns.heatmap(data=df_topics_trend_perc.transpose(),
                cmap=color_heatmap,
                cbar=False,
                square=True,
                annot=True,
                fmt='.1%')
    plt.title('Average change in the period 2021-2022 per model / topic')
    plt.xlabel('Topic Number')
    plt.savefig('figure_10_heatmap_topic_trends')
    plt.show()

    # Keyword match analysis between the models
    # Initial maximum value
    max_word_match = -1

    # Keyword match: KMeans - LDA
    keyword_similarity_kmeans_lda = pd.DataFrame(index=range(1, kmeans_topics+1), columns=range(1, lda_topics+1))
    keyword_similarity_kmeans_lda = keyword_similarity_kmeans_lda.astype('float')

    for kmeans_topic in range(1, kmeans_topics+1):
        kmeans_words = df[df['KMeans Topic'] == kmeans_topic]['KMeans Keywords'].iloc[0]

        for lda_topic in range(1, lda_topics+1):
            lda_words = df[df['LDA Topic'] == lda_topic]['LDA Keywords'].iloc[0]

            word_match = (2*n_top_words)-len(set(kmeans_words).union(set(lda_words)))
            keyword_similarity_kmeans_lda.loc[kmeans_topic, lda_topic] = word_match

            if word_match > max_word_match:
                max_word_match = word_match

    # Keyword match: KMeans - Top2Vec
    keyword_similarity_kmeans_top2vec = pd.DataFrame(index=range(1, kmeans_topics+1),
                                                     columns=range(1, top2vec_topics+1))
    keyword_similarity_kmeans_top2vec = keyword_similarity_kmeans_top2vec.astype('float')
    for kmeans_topic in range(1, kmeans_topics+1):
        kmeans_words = df[df['KMeans Topic'] == kmeans_topic]['KMeans Keywords'].iloc[0]

        for top2vec_topic in range(1, top2vec_topics+1):
            top2vec_words = df[df['Top2Vec (REDUCED) Topic'] == top2vec_topic]['Top2Vec (REDUCED) Keywords'].iloc[0]

            word_match = (2*n_top_words)-len(set(kmeans_words).union(set(top2vec_words)))
            keyword_similarity_kmeans_top2vec.loc[kmeans_topic, top2vec_topic] = word_match

            if word_match > max_word_match:
                max_word_match = word_match

    # Keyword match: LDA - Top2Vec
    keyword_similarity_lda_top2vec = pd.DataFrame(index=range(1, lda_topics+1), columns=range(1, top2vec_topics+1))
    keyword_similarity_lda_top2vec = keyword_similarity_lda_top2vec.astype('float')
    for lda_topic in range(1, lda_topics+1):
        lda_words = df[df['LDA Topic'] == lda_topic]['LDA Keywords'].iloc[0]

        for top2vec_topic in range(1, top2vec_topics+1):
            top2vec_words = df[df['Top2Vec (REDUCED) Topic'] == top2vec_topic]['Top2Vec (REDUCED) Keywords'].iloc[0]

            word_match = (2*n_top_words)-len(set(lda_words).union(set(top2vec_words)))
            keyword_similarity_lda_top2vec.loc[lda_topic, top2vec_topic] = word_match

            if word_match > max_word_match:
                max_word_match = word_match

    # Plotting keyword match analysis
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    sns.heatmap(data=keyword_similarity_kmeans_lda,
                cmap=sns.color_palette("rocket", as_cmap=True),
                cbar=False,
                square=True,
                vmax=max_word_match,
                xticklabels=2,
                yticklabels=2,
                ax=ax1)
    ax1.set_xlabel('LDA Topic Number')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.set_ylabel('KMeans Topic Number')

    sns.heatmap(data=keyword_similarity_kmeans_top2vec,
                cmap=sns.color_palette("rocket", as_cmap=True),
                cbar=False,
                square=True,
                vmax=max_word_match,
                xticklabels=2,
                yticklabels=2,
                ax=ax2)
    ax2.set_xlabel('Top2Vec Topic Number')
    ax2.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax2.set_ylabel('KMeans Topic Number')

    sns.heatmap(data=keyword_similarity_lda_top2vec,
                cmap=sns.color_palette("rocket", as_cmap=True),
                cbar=False,
                square=True,
                vmax=max_word_match,
                xticklabels=2,
                yticklabels=2,
                ax=ax3)
    ax3.set_xlabel('Top2Vec Topic Number')
    ax3.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax3.set_ylabel('LDA Topic Number')

    plt.suptitle('Keyword match analysis', fontsize=24, y=.64)
    plt.savefig('figure_11_heatmap_word_match', bbox_inches='tight')
    plt.show()

    kmeans_topic_1_words = df[df['KMeans Topic'] == 1]['KMeans Keywords'].iloc[0]
    kmeans_topic_2_words = df[df['KMeans Topic'] == 2]['KMeans Keywords'].iloc[0]
    kmeans_topic_3_words = df[df['KMeans Topic'] == 3]['KMeans Keywords'].iloc[0]
    lda_topic_1_words = df[df['LDA Topic'] == 1]['LDA Keywords'].iloc[0]
    top2vec_topic_6_words = df[df['Top2Vec (REDUCED) Topic'] == 6]['LDA Keywords'].iloc[0]
    top2vec_topic_9_words = df[df['Top2Vec (REDUCED) Topic'] == 9]['LDA Keywords'].iloc[0]

    wc_kmeans_topic_1 = WordCloud().generate(' '.join(kmeans_topic_1_words))
    plt.axis('off')
    plt.grid(False)
    plt.title('KMeans Topic 1 Keywords')
    plt.imshow(wc_kmeans_topic_1, interpolation="bilinear")
    plt.savefig('wordcloud_kmeans_topic_1')
    plt.show()

    wc_kmeans_topic_2 = WordCloud().generate(' '.join(kmeans_topic_2_words))
    plt.axis('off')
    plt.grid(False)
    plt.title('KMeans Topic 2 Keywords')
    plt.imshow(wc_kmeans_topic_2, interpolation="bilinear")
    plt.savefig('wordcloud_kmeans_topic_2')
    plt.show()

    wc_kmeans_topic_3 = WordCloud().generate(' '.join(kmeans_topic_3_words))
    plt.axis('off')
    plt.grid(False)
    plt.title('KMeans Topic 3 Keywords')
    plt.imshow(wc_kmeans_topic_3, interpolation="bilinear")
    plt.savefig('wordcloud_kmeans_topic_3')
    plt.show()

    wc_lda_topic_1 = WordCloud().generate(' '.join(lda_topic_1_words))
    plt.axis('off')
    plt.grid(False)
    plt.title('LDA Topic 1 Keywords')
    plt.imshow(wc_lda_topic_1, interpolation="bilinear")
    plt.savefig('wordcloud_lda_topic_1')
    plt.show()

    wc_top2vec_topic_6 = WordCloud().generate(' '.join(top2vec_topic_6_words))
    plt.axis('off')
    plt.grid(False)
    plt.title('Top2Vec Topic 6 Keywords')
    plt.imshow(wc_top2vec_topic_6, interpolation="bilinear")
    plt.savefig('wordcloud_top2vec_topic_6')
    plt.show()

    wc_top2vec_topic_9 = WordCloud().generate(' '.join(top2vec_topic_9_words))
    plt.axis('off')
    plt.grid(False)
    plt.title('Top2Vec Topic 9 Keywords')
    plt.imshow(wc_top2vec_topic_9, interpolation="bilinear")
    plt.savefig('wordcloud_top2vec_topic_9')
    plt.show()

    return


# ----- INITIAL VARIABLES AND SETTINGS -----
# Ignore PerformanceWarning from pd (Pandas) module
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

sns.set_style('whitegrid')

start_date = '2021-01-01'
end_date = '2022-12-01'

list_stop_words = set(stopwords.words('english'))
list_stop_words = list_stop_words.union({
    'one', 'two', 'three', 'paper', 'non', 'new', 'approach', 'left', 'show', 'also', 'based', 'results',
    'using', 'used', 'work', 'case', 'study', 'mathbb'
})
list_stop_words = list(list_stop_words)


# ----- MAIN CODE -----

# ----- PHASE 1: READ AND BASE ANALYSIS -----

# READ THE AVAILABLE DATA
df_dataset = read_dataset()

# READ THE AVAILABLE TAXONOMY
df_taxonomy = read_taxonomy()

print('\nStart analysis_basic for complete dataset')
analysis_basic(df_dataset)

print('\nStart preprocess_update_date')
df_dataset = preprocess_update_date(df_dataset)

print('\nStart filter_update_date')
df_dataset = filter_update_date(df_dataset, begin=start_date, end=end_date)

print('\nStart filter_duplicates')
df_dataset.drop_duplicates(subset='abstract', keep='last', inplace=True)

print('\nStart drop_columns')
df_dataset.drop(columns=['submitter', 'authors', 'comments', 'journal-ref', 'doi', 'versions', 'report-no', 'license'],
                inplace=True)

print('\nStart analysis_basic for filtered dataset')
analysis_basic(df_dataset)

print('\nStart preprocess_word_counts')
df_dataset = preprocess_word_counts(df_dataset)

print('\nStart preprocess_categories')
df_dataset = preprocess_categories(df_dataset, df_taxonomy)

print('\nStart analysis of word-counts')
analysis_word_counts(df_dataset)
print('\nStart analysis of taxonomy')
analysis_taxonomy(df_taxonomy)
print('\nStart analysis of categories')
analysis_categories(df_dataset, df_taxonomy)

print('\nStart NLTK Tokenizer')
df_dataset['processed_abstract'] = \
    df_dataset['abstract'].apply(lambda x: abstract_tokenzier_nltk(x, list_stop_words))
df_dataset['processed_abstract_tokenized'] = \
    df_dataset['processed_abstract'].apply(lambda text: text.split())

print('\nStart TF-IDF Vectorizer')
tfidf_matrix, tfidf_features = tfidf_vectorizer(df_dataset)
# ggf noch schauen, was genau als Input dienen soll

print('\nStart PCA Feature Reduction')
tfidf_matrix_reduced = pca_feature_reduction(tfidf_matrix)

print('\nStart KMeans Elbow Validation')
# kmeans_validation(tfidf_matrix_reduced, 8, 30)
# --> k=20

print('\nStart KMeans modeling')
kmeans_n_topics = 20
df_dataset = kmeans_modeling(df_dataset, tfidf_matrix_reduced, kmeans_n_topics)
# ggf noch schauen, was genau als Input dienen soll

print('\nStart assigning top_words')
df_dataset['KMeans Keywords'] = \
    df_dataset['KMeans Topic'].apply(lambda x: get_kmeans_top_words(x, tfidf_matrix, tfidf_features, n_top_words=20))

print('\nStart LDA Preoprocessing')
lda_dictionary, lda_corpus = lda_preprocessing(df_dataset)

'''print('\nStart LDA Validation')
lda_configuration_cv, lda_configuration_cnmpi = lda_validation(df_dataset,
                                                               lda_corpus,
                                                               lda_dictionary,
                                                               min_alpha=.6,
                                                               max_alpha=.7,
                                                               min_eta=.5,
                                                               max_eta=.6,
                                                               min_topics=8,
                                                               max_topics=24)
# --> k=21, alpha=0.65, eta=0.50
'''

print('\nStart LDA Modeling')
lda_n_topics = 21
lda_modeling(df_dataset, lda_corpus, lda_dictionary, lda_n_topics, alpha=.65, eta=.50)

print('\nStart Top2Vec Modeling')
top2vec_n_topics = max(lda_n_topics, kmeans_n_topics)
top2vec = top2vec_modeling(df_dataset, top2vec_n_topics)

print('\nStart Reassignin Topic Numbers')
df_dataset = reassign_topics(df_dataset, 'KMeans')
df_dataset = reassign_topics(df_dataset, 'LDA')
df_dataset = reassign_topics(df_dataset, 'Top2Vec (REDUCED)')

print('\nStart Topic Analysis')
analysis_topics(df_dataset, kmeans_n_topics, lda_n_topics, top2vec_n_topics, 20)

print('\nStart t-SNE Visualization of KMeans Clusters')
tsne_visualization(tfidf_matrix, df_dataset['KMeans Topic'], kmeans_n_topics)

print('\nStart saving dataset and taxonomy')
save_dataset(df_dataset)
save_taxonomy(df_taxonomy)
