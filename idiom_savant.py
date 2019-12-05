import process_data
import baseline_subtask2
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import os
import string
import time
from tqdm import tqdm # progress bar for long loops
from numpy.linalg import norm
from nltk import FreqDist
from nltk.corpus import wordnet as wn, stopwords, brown
from nltk.tokenize import RegexpTokenizer 
tokenizer = RegexpTokenizer(r'\w+') # tokeniser that removes punctuation

stopWords = set(stopwords.words('english'))

# retrieve the 100 most common words from the Brown Corpus, exclude stopwords and punctuation
words = brown.words()
filtered_words = [w.lower() for w in words if w.lower() not in stopWords and w not in string.punctuation and w not in "''``'--"]
word_freq = FreqDist(w for w in filtered_words)
most_common_words = set()
for wordtuple in list(word_freq.most_common(100)):
    most_common_words.add(wordtuple[0])


# this is a subset of only the 60000 most frequent words
# download from commit 45e9102bef1a699688bb2d62a9fa2551ed4b463c if needed
# glove_filename = os.path.join("embedding models", 'glove.6B.100d.60k.word2vec.txt')
# glove_model = KeyedVectors.load_word2vec_format(glove_filename, binary=False)

# full word2vec model trained by Google
# download: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
model_filename = os.path.join("embedding models", "GoogleNews-vectors-negative300.bin")
# takes about a minute to just create this model (it's big)
wv_model =  KeyedVectors.load_word2vec_format(model_filename, binary=True)

# sets for seeing which words are not in vocabulary
words_not_in_wordnet = set()
words_not_in_w2v = set()


def get_senses_of_word(word):
    """
    Return the senses of word from wordnet
    """
    word = wn.morphy(word) if wn.morphy(word) is not None else word
    synsets = wn.synsets(word)
    if len(synsets) == 0:
        words_not_in_wordnet.add(word)
        # print("{} not in WordNet".format(word))

    return synsets


def get_gloss_set(synset, include_examples=True):
    """
    return the definition as a set without stopwords or punctuation
    i.e. a group of unique words
    optionally get also examples from wordnet and include them in the set
    """
    gloss_tokens = set(tokenizer.tokenize(synset.definition())).difference(stopWords)

    if include_examples:
        for example in synset.examples():
            example = set(tokenizer.tokenize(example)).difference(stopWords)
            gloss_tokens = gloss_tokens.union(example)

    return gloss_tokens


def word_vector(word):
    """
    Return the word embedding of a word.
    """
    try:
        return wv_model[word]
    except KeyError as err:
        words_not_in_w2v.add(word)
        # print("Word2Vec error in word_vector():", err)


def similarity(w1, w2, metric=None, correction=True):
    """
    Return the similarity of two words.
    """
    sim = 0
    if metric == "cosine": # cosine similarity
        vector1 = word_vector(w1)
        vector2 = word_vector(w2)
        sim = np.inner(vector1, vector2) / (norm(vector1)*norm(vector2))
    
    else: # Gensim's built-in similarity metric, I don't know how it is computed
        try:
            sim = wv_model.similarity(w1, w2)
        except KeyError as err:
            # print("Word2Vec error in similarity():", err)
            sim = 0.2 # arbitrary
            # just for adding the words in the words_not_in_w2v set
            word_vector(w1) 
            word_vector(w2)
            
    # f_ws() correction function from the article
    if correction:
        if sim < 0.01:
            sim = 0
        else:
            sim = 1 - sim

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


def gloss_score_homographic(puns, use_context_gloss=False, use_examples=True, normalise=True):
    """
    Score each word according to Idiom Savant (p. 106):

    "[...] as additional context information, wi were replaced with set of gloss
    information extracted from its different senses, noted as gi, obtained from
    WordNet. While calculating similarity between gi and ci, two different strategies were
    employed.
    
    In the first strategy, the system computes similarities between every
    combination of gi and ci, and sum of similarity scores is the score for wi.
    
    In the second strategy, similarity score were calculated between gi and gj,
    the gloss of wj ∈ ci.
    
    [...] In puns, punning words and grounding words in context are often not adjacent.
    Thus the system does not consider the adjacent words of the candidate word.
    The system also ignored stopwords offered by NLTK."

    The article does not use examples from WordNet in addition to the definition.
    -> In this function examples are included optionally.

    The article does not describe any kind of normalisation of the score by the number
    of words that are included in the gloss (and examples). This makes the score higher
    for words that happen to include a long definition in WordNet.
    -> in this function the score is normalised by the number of similarities that are
    summed together

    POS damping:
    "In most of the cases, pun words and their grounding words in the context do not share the same
    part-of-speech (POS) tags. In the latter strategy, we added a POS damping factor of 0.2 
    if the POS tags of wi and wj are equal."

    Frequency damping:
    "We noticed that words with high frequency other than stopwords overshadow low
    frequency words since every word with high frequency poses certain similarity score with every
    other phrases. Thus we added a frequency damping factor(fij) of 0.1 to the score for whose words
    have frequencies more than 100 in Brown Corpus (Francis and Kucera, 1979)."
    """
    for punID, pun in tqdm(puns.items()): # looping ten puns takes about 30 seconds, so 2000 puns = 100 minutes 
        for wordID, word in pun.items():
            puns[punID][wordID]['score'] = 0 # deletes previous scores

            # exclude the word itself and adjacent words as described in the article
            exc = set(( wordID, wordID[:-1] + str(int(wordID[-1]) - 1), wordID[:-1] + str(int(wordID[-1]) + 1) ))
            # get the other tokens
            pun_tokens, pun_pos_tags = process_data.get_pun_tokens(pun, exclude=exc, return_pos_tags=True)

            # the senses from wordnet
            word_senses = get_senses_of_word(word['token'])
            for word_sense in word_senses:
                word_gloss = get_gloss_set(word_sense, include_examples=use_examples)
                len_word_gloss = len(word_gloss)

                # first strategy: compare word gloss to every context word
                if not use_context_gloss:
                    score =  sim_token_lists(word_gloss, pun_tokens)
                    if normalise:
                        score /= (len_word_gloss * len(pun_tokens))
                    puns[punID][wordID]['score'] +=score

                # second strategy: compare word gloss to the gloss of every context word
                elif use_context_gloss:
                    for context_word, context_word_pos in zip(pun_tokens, pun_pos_tags):
                        context_word_senses = get_senses_of_word(context_word)
                        # TODO: exclude senses with wrong POS
                        for context_word_sense in context_word_senses:
                            context_word_gloss = get_gloss_set(context_word_sense, include_examples=use_examples)
                            score = sim_token_lists(word_gloss, context_word_gloss)
                            if normalise:
                                score /= (len(context_word_gloss) * len_word_gloss)

                            puns[punID][wordID]['score'] += score

                    # POS damping
                    # this is pretty stupid in my opinion, can be skipped
                    # consider only two first chars e.g. VB == VBD, because "VBD"[:2] == "VB"
                    # if context_word_pos[:2] == word['pos'][:2]: 
                    #     puns[punID][wordID]['score'] *= 0.2

                    # Frequency damping
                    word_lemma = wn.morphy(word['token']) if wn.morphy(word['token']) is not None else word['token']
                    context_word_lemma = wn.morphy(context_word) if wn.morphy(context_word) is not None else context_word
                    if word_lemma in most_common_words:
                        # print('pun word "{}" is in most_common_words'.format(word['token']))
                        puns[punID][wordID]['score'] *= 0.1
                    elif context_word_lemma in most_common_words:
                        # print('context word "{}" is in most_common_words'.format(context_word))
                        puns[punID][wordID]['score'] *= 0.3


def get_results(puns):
    """
    Create results based on scores.
    """
    results = {} # baseline_subtask2.select_last_word_exclude_most_common(puns)
    for punID, pun in puns.items():
        best_score = 0
        for wordID, word in pun.items():
            if word['score'] > best_score:
                best_score = word['score']
                bestID = wordID

        results[wordID] = bestID

    return results


def print_scores(pun):
    # print(pun.values())
    print("pun:")
    for wordID, word in pun.items():
        print(wordID, word['token'], word['score'])


puns, taskID = process_data.get_puns()
puns = process_data.truncate_puns(puns, keep=20)
process_data.lowercase_caps_lock_words(puns)
process_data.add_pos_tags(puns)
puns = process_data.remove_stopwords(puns)
puns = process_data.only_content_words(puns)
process_data.lowercase(puns)
process_data.add_word_numbers(puns)
gloss_score_homographic(puns, use_context_gloss=True, use_examples=True, normalise=True)
results = get_results(puns)
process_data.write_results(results, filename=taskID + "-idiom_savant", timestamp=False)

print_scores(puns["hom_1"])
print_scores(puns["hom_2"])
print_scores(puns["hom_3"])
print_scores(puns["hom_4"])
print_scores(puns["hom_5"])

# for item in words_not_in_w2v:
#     print(item)

# for item in words_not_in_wordnet:
#     print("not in wordnet:", item)


def n_gram_prob(ngram):
    """
    search the n-gram probability (or count) from google n-grams
    """
    # return prob