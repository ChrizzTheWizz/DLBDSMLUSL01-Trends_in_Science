# Use-Case: Trends in Science
# Categorizing Scientific Trends with Unsupervised Machine Learning - A Case Study Based on the arXiv Dataset
# Project: HabitTracker

## Table of Contents
1. [About the project](#About-the-project)
2. [Requirements](#Requirements)
3. [Installation](#Installation)
4. [Usage / How to](#Usage-/-How-to)
5. [Contributing](#Contributing)


## About the project
Based on the arXiv dataset, the most important topics / trends from 2021-2022 are extracted using KMeans, LDA (gensim) and Top2Vec.

IMPORTANT:
This app was developed as part of a case-study course for my Data Science degree and contains the basic requirements according to the assignment. Further development is not planned, although many improvements concerning object-oriented programming and error-handling would be possible and within the scope of my skills.

## Requirements

**General:** 
This app was written in Python. Make sure you have Python 3.8+ installed on your device. 
You can download the latest version of Python [here](https://www.python.org/downloads/). 

**Packages:**
* [numpy](https://numpy.org) (install via "pip install numpy")
* [pandas](https://pandas.pydata.org/about/index.html) (install via "pip install pandas")
* [nltk](https://github.com/nltk/nltk) (install via "pip install nltk")
* [seaborn](https://github.com/mwaskom/seaborn) (install via "pip install seaborn")
* [gensim](https://github.com/RaRe-Technologies/gensim) (install via "pip install gensim")
* [yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) (install via "pip install yellowbrick")
* [top2vec](https://github.com/ddangelov/Top2Vec) (install via "pip install top2vec")
* [wordcloud](https://pypi.org/project/wordcloud/) (install via "pip install wordcloud")
* [pyLDAvis](https://github.com/bmabey/pyLDAvis) (install via "pip install pyldavis")
* [sklearn](https://scikit-learn.org/stable/) (install via "pip install sklearn")


## Installation

**How To:**<br>
The easiest way to use this app is to download the following files:

/habittracker/main.py

arXiv Dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv

## Usage / How to

You only have to run main.py. Make sure you have downloaded the arXiv Dataset.
There are a few parameters set in the code. Feel free to change those - for example the filter criteria for the time period or the stop-words.  

## Contributing 
With reference to the fact that this app was created in the course of my studies and I am therefore in a constant learning process, I am happy to receive any feedback.
So please feel free to contribute pull requests or create issues for bugs and feature requests.
