import xml.etree.ElementTree as ET
import os
import time
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.util import ngrams

DATA_DIR    = os.path.join(".", "semeval2017_task7", "data", "test")
RESULTS_DIR = os.path.join(".", "results")

def get_puns(subtask=2, h="homographic"):
    """
    Create a dictionary nested in a dictionary that contains all puns
    of one subtask, either homographic or heterographic.
    
    To get a specific word in a (possible) pun from the nested dictionary:
    puns["hom_2250"]["hom_2250_1"]['token']

    return also the subtask ID, e.g. "subtask2-homographic"
    """
    filename = "subtask" + str(subtask) + "-" + h + "-test" + ".xml"
    root = ET.parse(os.path.join(DATA_DIR, filename)).getroot()

    puns = {}
    for pun in root:
        puns[pun.attrib['id']] = {}
        for word in pun:
            puns[pun.attrib['id']][word.attrib['id']]           = {}
            puns[pun.attrib['id']][word.attrib['id']]['token']  = word.text

    taskID = root.attrib['id']
    return puns, taskID


def get_all_puns():
    """
    Create a three-fold nested dictionary to get all the puns of all subtasks.

    To get a specific word in a (possible) pun from the nested dictionary:
    all_puns["subtask2-homographic"]["hom_2250"]["hom_2250_4"]
    """
    all_puns = {}
    for task in range(1,4):
        puns, taskid = get_puns(subtask=task, h="homographic")
        all_puns[taskid] = puns
        puns, taskid = get_puns(subtask=task, h="heterographic")
        all_puns[taskid] = puns
    return all_puns


def get_pun_tokens(pun, exclude=[], return_pos_tags=False):
    """
    Returns a list of the words (tokens) from a pun dictionary.
    """
    pun_tokens = []
    pos_tags = []
    for wordID, word in pun.items():
        if wordID not in exclude:
            pun_tokens.append(word['token'])
            if return_pos_tags:
                pos_tags.append(word['pos'])

    if return_pos_tags:
        return pun_tokens, pos_tags

    return pun_tokens


def truncate_puns(puns, keep=10):
    """
    Truncate puns dictionary for evaluation purposes.
    """
    truncated_puns = {}
    i = 0
    for punID, pun in puns.items():
        truncated_puns[punID] = pun
        i += 1
        if keep == i: break

    return truncated_puns


def write_results(results, filename="results", timestamp=True):
    """
    Write results into a text file from the results dictionary.
    """
    if timestamp:
        filename += time.strftime("%Y%m%d-%H%M%S")
    filename += ".txt"

    with open(os.path.join(RESULTS_DIR, filename), "w") as f:
        for k in results.keys():
            f.write(str(k) + " " + str(results[k]) + "\n")


def remove_punctuation(puns):
    """
    Remove punctuation characters from a puns dictionary.
    This is unnecessary if you use only_content_words(puns).
    """
    new_puns = {}
    for punID, pun in puns.items():
        new_puns[punID] = {}
        for wordID, word in pun.items():
            if word['token'] not in string.punctuation:
                new_puns[punID][wordID] = word

    return new_puns


def remove_stopwords(puns):
    """
    Remove stop words from a puns dictionary.
    This is unnecessary if you use only_content_words(puns).
    """
    stopWords = set(stopwords.words('english'))
    new_puns = {}
    for punID, pun in puns.items():
        new_puns[punID] = {}
        for wordID, word in pun.items():
            if word['token'] not in stopWords:
                new_puns[punID][wordID] = word

    return new_puns


def lowercase_caps_lock_words(puns):
    """
    There are some words WRITTEN WITH CAPS LOCK so make those lowercase.
    A capitalised Word (only first letter is upper case) should not be affected.
    Ignore one-letter words.
    """
    for punID, pun in puns.items():
        for wordID, word in pun.items():
            if word['token'].isupper() and len(word['token']) > 1:
                word['token'] = word['token'].lower()


def lowercase(puns):
    """
    Turn all words except proper nouns into lowercase.
    """
    for punID, pun in puns.items():
        for wordID, word in pun.items():
            if word['pos'] not in {'NNP', 'NNPS'}:
                word['token'] = word['token'].lower()


def add_pos_tags(puns):
    """
    Add POS tags in the puns dictionary.
    The words are dictionaries with keys 'token' and 'pos'
    e.g. 'hom_1_8': {'token': 'sauna', 'pos': 'NN'}

    NLTK Part of Speech tags from
    https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/ :
    CC      coordinating conjunction
    CD      cardinal digit
    DT      determiner
    EX      existential there (like: "there is" ... think of it like "there exists")
    FW      foreign word
    IN      preposition/subordinating conjunction
    JJ      adjective 'big'
    JJR     adjective, comparative 'bigger'
    JJS     adjective, superlative 'biggest'
    LS      list marker 1)
    MD      modal could, will
    NN      noun, singular 'desk'
    NNS     noun plural 'desks'
    NNP     proper noun, singular 'Harrison'
    NNPS    proper noun, plural 'Americans'
    PDT     predeterminer 'all the kids'
    POS     possessive ending parent's
    PRP     personal pronoun I, he, she
    PRP$    possessive pronoun my, his, hers
    RB      adverb very, silently,
    RBR     adverb, comparative better
    RBS     adverb, superlative best
    RP      particle give up
    TO      to go 'to' the store.
    UH      interjection 'errrrrrrrm'
    VB      verb, base form take
    VBD     verb, past tense took
    VBG     verb, gerund/present participle taking
    VBN     verb, past participle taken
    VBP     verb, sing. present, non-3d take
    VBZ     verb, 3rd person sing. present takes
    WDT     wh-determiner which
    WP      wh-pronoun who, what
    WP$     possessive wh-pronoun whose
    WRB     wh-abverb where, when
    """
    for punID, pun in puns.items():
        pun_tokens = get_pun_tokens(pun)
        postags = pos_tag(pun_tokens)
        for wordID, posItem in zip(pun.keys(), postags):
            puns[punID][wordID]['pos'] = posItem[1]


def add_word_numbers(puns):
    """
    Add the number from wordID as a key-value pair in the word dictionary.
    E.g. "hom_345_13" -> word["word_number"]: 13 
    """
    for punID, pun in puns.items():
        for wordID, word in pun.items():
            _counter = 0
            for char_ind in range(len(wordID)):
                if wordID[char_ind] == "_":
                    _counter += 1
                if _counter == 2:
                    word['word_number'] = int(wordID[char_ind + 1:])
                    break


def only_content_words(puns):
    """
    Keep only nouns, verbs, adverbs and adjectives in the puns dictionary.
    Assumes puns dictionary includes POS tags.
    """
    content_tags = ['FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    new_puns = {}
    for punID, pun in puns.items():
        new_puns[punID] = {}
        for wordID, word in pun.items():
            if word['pos'] in content_tags:
                new_puns[punID][wordID] = word

    return new_puns


def get_trigrams(puns):
    """
    Separate the context into trigrams
    Return a list of tuples for each pun: [(word1, word2, word3), (word2, word3, word4)]
    """
    trigrams = {}
    for punID, pun in puns.items():
        tokenized_sent = get_pun_tokens(pun)
        trigrams[punID] = list(ngrams(tokenized_sent, 3))

    return trigrams

