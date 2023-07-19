import pickle
import nltk
import re
from gensim.parsing.preprocessing import remove_stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, LancasterStemmer
from nltk.stem.snowball import EnglishStemmer


tag_re = re.compile(r'<[^>]+>')

def remove_tags(text):
    return tag_re.sub('', text)

def preprocess_text(text):
    lemma = EnglishStemmer()
    #remove urls
    text = ' '.join(word for work in text.split() if not word.startswith('http'))
    text = ' '.join(word for work in text.split() if not word.startswith('www'))
    # remove punctions and numbers
    text = re.sub('[^a-zA-Z]', ' ',text)
    # remove single character
    text = re.sub(r"\s+[a-zA-Z]\s+",' ',text)
    # remove @username
    text = ' '.join(word for word in text.split() if not word.startswith('@'))
    #remove hashtags
    text = ' '.join(word[1:] if word.startswith('#') else word for word in text.split())
    # remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    #remove stopsword
    text = remove_stopwords(text.lower())
    