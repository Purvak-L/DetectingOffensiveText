import re, string, unicodedata
import nltk
import contractions
import inflect
import string
import unicodedata
import re
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import pandas as pd

class PreprocessingPipeline:
    
    def __init__(self):
        self.document = 'document'
    
    def remove_strip_links(self,text):
        link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
        links         = re.findall(link_regex, text)
        for link in links:
            text = text.replace(link[0], ', ')    
        return text
    
    def strip_mentions_hashtags(self, text):
        entity_prefixes = ['@','#']
        for separator in  string.punctuation:
            if separator not in entity_prefixes :
                text = text.replace(separator,' ')
        words = []
        for word in text.split():
            word = word.strip()
            if word:
                if word[0] not in entity_prefixes:
                    words.append(word)
        return ' '.join(words)
    
    def remove_special_characters(text, remove_digits=False):
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        #text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text
        
    def remove_non_ascii(self, words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words.split():
            word = word.strip()
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return ' '.join(new_words)

    def to_lowercase(self, words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words.split():
            word = word.strip()
            new_word = word.lower()
            new_words.append(new_word)
        return ' '.join(new_words)

    def remove_punctuation(self, words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words.split():
            word = word.strip()
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return ' '.join(new_words)

    def replace_numbers(self, words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words.split():
            word = word.strip()
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return ' '.join(new_words)

    def remove_stopwords(self,words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words.split():
            word = word.strip()
            if word not in stopwords.words('english'):
                new_words.append(word)
        return ' '.join(new_words)

    def stem_words(self, words):
        """Stem words in list of tokenized words"""
        stemmer = LancasterStemmer()
        stems = []
        for word in words.split():
            word = word.strip()
            stem = stemmer.stem(word)
            stems.append(stem)
        return ' '.join(stems)

    def lemmatize_verbs(self, words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words.split():
            word = word.strip()
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return ' '.join(lemmas)

    def normalize(self, words):
        words = self.remove_strip_links(words)
        words = self.strip_mentions_hashtags(words)
        words = self.remove_non_ascii(words)
        words = self.to_lowercase(words)
        words = self.remove_punctuation(words)
        words = self.replace_numbers(words)
        words = self.remove_stopwords(words)
        return words


if __name__ == '__main__':
    instance = PreprocessingPipeline()
    df_new = pd.read_csv('hatespeech.csv')
    hs = pd.read_csv('hatespeech.csv', encoding="ISO-8859-1",index_col=6, keep_default_na=False)
    #print(hs.head())

    orig = pd.read_csv('NAACL_SRW_2016.csv', index_col=0, header=None)
    orig.index.name = 'ID'
    orig = orig.rename(columns={1: 'Class'})
    orig.index = orig.index.astype(str)
    #print(orig.head())

    #merging the two dataframes
    hs = pd.merge(hs, orig, how='inner', left_index=True, right_index=True)
    #print(hs.head())
    df_new = hs
    df_new = df_new.dropna()
    df_new = df_new[['Tweets','Class']]

#     df = pd.read_csv('new_hatespeech_processed.csv',encoding="ISO-8859-1")
#     df_new = df[['tweet_id','does_this_tweet_contain_hate_speech','tweet_text']]
#     df_new = df_new.dropna()
#     df_new.columns = ['ID','label','data']

    #df_new.columns = ['data','label']
    df_new['data'] = df_new['data'].apply(lambda x: instance.normalize(x))
    df_new.to_csv('new_hatespeech_processed.csv')

    print("Done!")
