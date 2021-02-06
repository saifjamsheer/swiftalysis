#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:02:24 2020

@author: Saif
"""

import pandas as pd
import ast
import scipy
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import OrderedDict
from operator import itemgetter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import math
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer

class TopicModeler():
    
    def show_topics(self, model, num_topics, n_top_words, feat_names):
    
        word_dict = {}
        H = model.components_
        
        for i in range(num_topics):
            
            words_ids = H[i].argsort()[:-n_top_words-1:-1]
            words = [feat_names[key] for key in words_ids]
            word_dict['Topic' + '{:2d}'.format(i+1)] = words;
        
        return pd.DataFrame(word_dict);
    
    def topics(self, dataset):
        
        matrix = scipy.sparse.csr_matrix(dataset.values)
        
        n_topics = 3
        n_words = 10
        
        model = NMF(n_components=n_topics, init='nndsvd').fit(matrix)
        W = model.fit_transform(matrix)
        
        feat_names = dataset.columns.values.tolist()
        datatopic = self.show_topics(model, n_topics, n_words, feat_names)
        
        return datatopic, W
        
    def tfidf(self, dataset):
        
        df = dataset.copy()
        wanalyzer = WordAnalyzer()
        
        words = wanalyzer.unique_words(df)
        
        df['wdict'] = df['lyrics'].apply(lambda x: self.computeF(words, x))
        df['tf'] = df.apply(lambda x: self.computeTF(x.wdict, x.lyrics), axis=1)
        
        idf = self.computeIDF(df)
        
        df['tfidf'] = df['tf'].apply(lambda x: self.computeTFIDF(x, idf))
        
        df = df.drop(columns=['lyrics', 'wdict', 'tf'])
        
        df = pd.DataFrame(df['tfidf'].values.tolist(), index=df.index)
        
        return df
    
    def computeF(self, unique, lyrics):
        
        wdict = dict.fromkeys(unique, 0)
        
        for word in lyrics:
            wdict[word] += 1
        
        return wdict
    
    def computeTF(self, wd, lyrics):
        
        tf = {}
        
        num_words = len(lyrics)
        
        for word, count in wd.items():
            tf[word] = count / float(num_words)
        
        return tf
    
    def computeIDF(self, dataset):
        
        N = dataset.shape[0]
        df = dataset.copy()
        
        documents = df['wdict']
        idf = dict.fromkeys(documents.iloc[0].keys(), 0)
        
        for doc in documents:
            for word, val in doc.items():
                if val > 0:
                    idf[word] += 1
    
        for word, val in idf.items():
            idf[word] = math.log(N / float(val))
            
        return idf
    
    def computeTFIDF(self, tf, idf):
        
        tfidf = {}
        
        for word, val in tf.items():
            tfidf[word] = val * idf[word]
        
        return tfidf
    
    def run(self, dataset):
        
        dataset = dataset['fearless']
        
        df = self.tfidf(dataset)
        dfT, W = self.topics(df)
        
        dfW = pd.DataFrame(W)
        threshold = 0.05
        dfW = dfW.transform(lambda x: x > threshold)
        
        dfTopics = pd.merge(dataset['album'], dfW, left_index=True, right_index=True)

class WordAnalyzer():
    
    def total_words(self, dataset):
        
        """
        Returns a new dataset with a column indicating the total 
        number of words in a song
        """
        
        df = dataset.copy()
        df['length'] = df['lyrics'].apply(lambda x: len(x)) 
        df = df.drop(columns=['album', 'lyrics'])
        
        return df
    
    def avg_words(self, dataset):
        
        """
        Returns the average number of words in an album
        """
        
        return int(dataset['length'].mean())
    
    def unique_words(self, dataset):
        
        """
        Returns the number of unique words in an album
        """
        
        df = dataset.copy()
        words = []
        
        for index, row in df.iterrows():
            words.extend(row['lyrics'])
        
        unique_words = list(set(words))
        
        return unique_words
    
    def get_frequencies(self, lyrics):
        
        """
        Returns a dictionary containing the number of occurrences 
        of all words in an album
        """
        
        w = {}
    
        for lyric in lyrics:
            for word in lyric:
                if word not in w.keys():
                    w[word] = 1
                else:
                    w[word] += 1
        
        frequencies = OrderedDict(sorted(w.items(), key=itemgetter(1), reverse=True))
        
        return frequencies
    
    def word_cloud(self, frequencies):
        
        """
        Generates a word cloud based on the frequencies of
        the words in an album
        """
                
        wc = WordCloud(background_color='white')
        wc.generate_from_frequencies(frequencies)
        
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()
    
    def top_n_unigrams(self, frequencies, n):
        
        """
        Returns the ten words with the highest frequency
        count in an album
        """
        
        top_n = dict(list(frequencies.items())[:n])
        
        return top_n
    
    def top_n_bigrams(self, dataset, n):
        
        df = dataset.copy()
        df['lyrics'] = df['lyrics'].apply(lambda x: ' '.join(x))
        corpus = []
        
        for index, row in df.iterrows():
            
            lyrics = row['lyrics']
            corpus.append(lyrics)
        
        vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        
        bigrams = []
        
        for item in words_freq.copy():
            
            bigram = item[0]
            words = bigram.split()
            
            if len(words) == len(set(words)):
                bigrams.append(item)
        
        return bigrams[:n]
    
    def top_n_trigrams(self, dataset, n):
        
        df = dataset.copy()
        df['lyrics'] = df['lyrics'].apply(lambda x: ' '.join(x))
        corpus = []
        
        for index, row in df.iterrows():
            
            lyrics = row['lyrics']
            corpus.append(lyrics)
        
        vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        
        trigrams = []
        
        for item in words_freq.copy():
            
            trigram = item[0]
            words = trigram.split()
            
            if len(words) == len(set(words)):
                trigrams.append(item)
        
        return trigrams[:n]
    
    def run(self, dfo, dfc, o, c):
        
        #words_set = analyzer.total_words(o['folklore'])
        #avg_length = analyzer.avg_words(words_set)
        #num_unique = analyzer.unique_words(o['folklore'])
        #n_bigrams = self.top_n_bigrams(c['folklore'], 30)
        #n_trigrams = self.top_n_trigrams(c['folklore'], 30)
        
        return 1

class SentimentClassifier():

    def classify(self, lyric):
        
        """
        Classifies a song as either positive or negative
        """
        
        sid = SentimentIntensityAnalyzer()
        c = 0
        
        for sentence in lyric:
            score = sid.polarity_scores(sentence)
            compound = score['compound']
            c += compound
        
        c /= len(lyric)
        
        if c > 0:
            return 'pos'
        else:
            return 'neg'
    
    def polarity(self, dataset):
        
        """
        Returns a new dataset with a column indicating the 
        polarity of each song on an album
        """
        
        df = dataset.copy()
        df['polarity'] = df['lyrics'].apply(lambda x: self.classify(x)) 
        df = df.drop(columns=['album', 'lyrics'])
        
        return df
    
    def run(self, dataset, d):
        
       df_test = self.polarity(d['folklore'])
       pos = df_test.query('polarity == "pos"').polarity.count()
       neg = df_test.query('polarity == "neg"').polarity.count()
    
        return 1
        
def convert(stringified): 
    
    wordlist = ast.literal_eval(stringified)
    
    return wordlist

def datamod(dataset):
    
    df = dataset.copy()
    d = {}
    
    df['lyrics'] = df['lyrics'].apply(lambda x: convert(x))
    
    albums = {k: v for k, v in df.groupby('album')}
    df.set_index(keys=['album'], drop=False,inplace=True)
    albums = df['album'].unique().tolist()
    df.reset_index(inplace=True, drop=True)
    
    for album in albums:
    
        data = df.loc[df.album == album]
    
        d.update({
                album.lower(): data
                })
    
    return [df, d]

def main():
    
    modeler = TopicModeler()
    classifier = SentimentClassifier()
    analyzer = WordAnalyzer()
    
    dfs = pd.read_csv('sentence-lyrics.csv', index_col=0)
    [dfs, s] = datamod(dfs)
    
    classifier.run(dfs, s)
    
    """
    DATASET 1: ORIGINAL SET OF LYRICS
    """
        
    dfo = pd.read_csv('original-lyrics.csv', index_col=0)
    [dfo, o] = datamod(dfo)
    
    """
    DATASET 2: CLEANED SET OF LYRICS
    """
    
    dfc = pd.read_csv('cleaned-lyrics.csv', index_col=0)
    [dfc, c] = datamod(dfc)
    
    analyzer.run(dfo, dfc, o, c)
    modeler.run(c)
    
if __name__ == "__main__":
    main()
