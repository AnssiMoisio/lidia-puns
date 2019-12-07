import numpy as np
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

stopWords = set(nltk.corpus.stopwords.words('english'))

# retrieve the n_exclude most common words from the Brown Corpus, exclude stopwords and punctuation
words = nltk.corpus.brown.words()
filtered_words = [w.lower() for w in words if w.lower() not in stopWords and w not in string.punctuation and w not in "''``'--"]

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def get_most_similar_word(puns):
    puns = nltk.word_tokenize(puns)
    similar_words = {}
    prondict = nltk.corpus.cmudict.dict()
    filtered_prons = []
    pun_prons =[]

    for word in puns:
        try:
            pun_prons.append(prondict[word])
        except KeyError:
            pass

    for i in range(len(filtered_words)):
        try:
            filtered_prons.append(prondict[filtered_words[i]])
        except KeyError:
            pass

    for word in pun_prons:
        for i in range(len(filtered_prons)):
            distance = int(levenshtein(word, filtered_prons[i]))
            if distance <= 1:
                similar_words[word] = (filtered_words[i], distance)

    print(similar_words)


def get_bigrams():
    with open('output.txt', 'r') as f:
        l = f.read().splitlines()

    l = [elem.replace(',', '') for elem in l]

    #tokenizer = TweetTokenizer
    l = [word_tokenize(elem) for elem in l]


    n_grams = []
    for elem in l:
        n_grams.append(set(ngrams(elem, 2)))

    n_grams = [j for i in n_grams for j in i]

    return n_grams

import process_data
puns, taskID = process_data.get_puns()

tokens = []
for punID, pun in puns.items():
    tokens.append(process_data.get_pun_tokens(pun))

tokens = [j for i in tokens for j in i]
count = 0
bigrams = get_bigrams()
print(len(set(bigrams)))
out = [item for t in bigrams for item in t]
print(len(set(out).intersection(tokens)))