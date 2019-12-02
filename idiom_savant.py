import process_data
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from numpy.linalg import norm
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

# tutorial for gensim & word2vec:
# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
# this is a subset of only the 60000 most frequent words
# (a full model should be downloaded for the final version)
glove_filename = 'glove.6B.100d.60k.word2vec.txt'
glove_model = KeyedVectors.load_word2vec_format(glove_filename, binary=False)


def get_senses_of_word(word):
    """
    Return the senses of word from wordnet
    """
    # lemmatize word
    # word = wn.morphy(word) if wn.morphy(word) is not None else word
    return wn.synsets(word)


def word_vector(word):
    """
    Return the word embedding of a word.
    """
    return glove_model[word]


def similarity(w1, w2, metric=None):
    """
    Return the similarity of two words.
    """
    sim = 0
    if metric == "cosine":
        vector1 = word_vector(w1)
        vector2 = word_vector(w2)
        sim = np.inner(w1, w2) / (norm(w1)*norm(w2))
    else:
        try:
            sim = glove_model.similarity(w1.lower(), w2.lower())
        except KeyError as err:
            print(err)

    return sim


def sim_token_lists(lista, listb):
    """
    Calculate similarity of two lists of tokens.
    """
    similarity_sum = 0
    for a in lista:
        for b in listb:
            similarity_sum += similarity(a, b)

    return similarity_sum


def baseline_score_homographic(puns):
    """
    Select the word with the highest similarity with the context.

    "In the baseline model we design, the pun potential score of a word wi
    is computed as the sum of cosine similarities between the word wi and every
    word in context wj ∈ ci. The word with highest score is returned as the punning
    word." p. 106 Idiom Savant
    """
    for punID, pun in puns.items():
        pun_tokens = process_data.get_pun_tokens(pun)
        for wordID, word in pun.items():
            puns[punID][wordID]['score'] = sim_token_lists([word['token']], pun_tokens)

    return puns


def gloss_score_homographic(puns, use_context_gloss=False):
    """
    Score each word according to Idiom Savant (p. 106):

    "[...] as additional context information, wi were replaced with set of gloss
    information extracted from its different senses, noted as gi, obtained from
    WordNet. While calculating similarity between gi and ci, two different strategies were
    employed.
    
    In the first strategy, the system computes similarities between every
    combination of gi and ci, and sum of similarity scores is the score for wi.
    
    In the second strategy, similarity score were calculated between gi and gj,
    the gloss of wj ∈ ci."
    """
    for punID, pun in puns.items():
        for wordID, word in pun.items():
            puns[punID][wordID]['score'] = 0
            pun_tokens = process_data.get_pun_tokens(pun, exclude=set([wordID]))
            word_senses = get_senses_of_word(word['token'])
            for word_sense in word_senses:
                word_gloss = tokenizer.tokenize(word_sense.definition())

                # second strategy
                if use_context_gloss:
                    for context_word in pun_tokens:
                        context_word_senses = get_senses_of_word(context_word)
                        for context_word_sense in context_word_senses:
                            context_word_gloss = tokenizer.tokenize(context_word_sense.definition())
                            puns[punID][wordID]['score'] += sim_token_lists(word_gloss, context_word_gloss)

                # first strategy
                elif not use_context_gloss:
                    puns[punID][wordID]['score'] += sim_token_lists(word_gloss, pun_tokens)

    return puns
                    
"""
puns, taskID = process_data.get_puns()
puns = process_data.truncate_puns(puns)
puns = process_data.add_pos_tags(puns)
puns = process_data.only_content_words(puns)
r = gloss_score_homographic(puns, use_context_gloss=True)
print(puns["hom_3"])
"""

def n_gram_prob(ngram):
    """
    search the n-gram probability (or count) from google n-grams
    """
    # return prob
