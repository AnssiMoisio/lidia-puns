import numpy as np
import nltk
import string
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams
import process_data

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


def get_sound_pairs():
    with open('output.txt', 'r') as f:
        l = f.read().splitlines()

    tokenizer = TweetTokenizer()
    l = [elem.replace(',', '') for elem in l]
    l = [tokenizer.tokenize(elem) for elem in l]

    sound_dict = {}
    #sound_pairs = []
    from itertools import permutations

    for elem in l:
        sound_dict[elem[0]] = list(permutations(elem, 2))
        #print("added:"+elem[0] + " as key and ", list(permutations(elem, 2)), "as value")
        sound_dict[elem[1]] = list(permutations(elem, 2))
        #print("added:"+elem[1] + " as key and ", list(permutations(elem, 2)), "as value")
        if len(elem) == 3:
            sound_dict[elem[2]] = list(permutations(elem, 2))
            #print("added:"+elem[2] + " as key and ", list(permutations(elem, 2)), "as value")
        elif len(elem) == 4:
            #print(elem)
            #print("addedhiiiii:"+elem[3] + " as key and ", list(permutations(elem, 2)), "as value")
            sound_dict[elem[3]] = list(permutations(elem, 2))
            sound_dict[elem[2]] = list(permutations(elem, 2))

        #sound_pairs.append(set(ngrams(elem, 2)))
    #sound_pairs = [j for i in sound_pairs for j in i]
    #sound_dict = dict(sound_pairs)
    from itertools import chain

    for k in sound_dict:
        sound_dict[k] = list(chain(*sound_dict[k]))
        sound_dict[k] = set(sound_dict[k])
    for key, value in sound_dict.items():
        if key in value:
            value.remove(key)
    print(sound_dict)
    return sound_dict

puns, taskID = process_data.get_puns()

tokens = []
for punID, pun in puns.items():
    tokens.append(process_data.get_pun_tokens(pun))

tokens = [j for i in tokens for j in i]

sound_pairs = get_sound_pairs()
#print(len(set(sound_pairs)))
out = [item for t in sound_pairs for item in t]
#print(len(set(out).intersection(tokens)))
