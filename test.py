from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

# read pdfs (folder: Wortprotokolle) content to docs string list
import os
import fitz  # PyMuPDF
import re
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

german_stop_words = stopwords.words('german')
vectorizer_model = CountVectorizer(stop_words=german_stop_words)


# get all pdfs in folder
pdf_folder = "Wortprotokolle"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# read all pdfs
docs = []

for pdf_file in pdf_files:
    # open pdf
    pdf_path = os.path.join(pdf_folder, pdf_file)
    doc = fitz.open(pdf_path)

    # read all pages
    for page in doc:
        text = page.get_text()
        docs.append(text)

    doc.close()

topic_model = BERTopic(language="german", verbose=True,
                       vectorizer_model=vectorizer_model)

topic, prob = topic_model.fit_transform(docs)

print(topic_model.get_topic_info())

topic_model.visualize_topics()
topic_model.visualize_barchart()
