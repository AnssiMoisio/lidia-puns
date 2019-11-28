import xml.etree.ElementTree as ET
import os
import time
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

DATA_DIR    = os.path.join(".", "semeval2017_task7", "data", "test")
RESULTS_DIR = os.path.join(".", "results")

def get_puns(subtask=2, h="homographic"):
    """
    Create a dictionary nested in a dictionary that contains all puns
    of one subtask, either homographic or heterographic.
    
    To get a specific word in a (possible) pun from the nested dictionary:
    puns["hom_2250"]["hom_2250_1"]

    return also the subtask ID, e.g. "subtask2-homographic"
    """
    filename = "subtask" + str(subtask) + "-" + h + "-test" + ".xml"
    root = ET.parse(os.path.join(DATA_DIR, filename)).getroot()

    puns = {}
    for pun in root:
        puns[pun.attrib['id']] = {}
        for word in pun:
            puns[pun.attrib['id']][word.attrib['id']] = word.text

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


def write_results(results, filename="results", timestamp=True):
    """
    Write results into a text file from a list of tuples.
    """
    if timestamp:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename += timestr
    filename += ".txt"

    with open(os.path.join(RESULTS_DIR, filename), "w") as f:
        for r in results:
            f.write(str(r[0]) + " " + str(r[1]) + "\n")


def remove_punctuation(puns):
    """
    Remove punctuation characters from a puns dictionary.
    This is unnecessary if you use only_content_words(puns).
    """
    new_puns = {}
    for punID, pun in puns.items():
        new_puns[punID] = {}
        for wordID, word in pun.items():
            if word not in string.punctuation:
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
            if word not in stopWords:
                new_puns[punID][wordID] = word

    return new_puns


def add_pos_tags(puns):
    """
    Add POS tags in the puns dictionary.
    The words are tuples (word, POS_tag), e.g. 'hom_1_8': ('sauna', 'NN')

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
    new_puns = {}
    for punID, pun in puns.items():
        new_puns[punID] = {}
        pun_tokens = list(pun.values())
        postags = pos_tag(pun_tokens)
        for wordID, posItem in zip(pun.keys(), postags):
            new_puns[punID][wordID] = posItem

    return new_puns


def only_content_words(puns):
    """
    Keep only nouns, verbs, adverbs and adjectives in the puns dictionary.
    Assumes puns dictionary includes POS tags.
    """
    content_tags = ['FR', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'RB', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    new_puns = {}
    for punID, pun in puns.items():
        new_puns[punID] = {}
        for wordID, word in pun.items():
            if word[1] in content_tags:
                new_puns[punID][wordID] = word

    return new_puns


def get_trigrams(puns):
    """
    Separate the context into trigrams
    Return a list of tuples: (word1, word2, word3)
    """
    tokenized_sents = []
    trigrams = []
    stop_words = set(stopwords.words('english'))

    for punID, pun in puns.items():
        for wordID, word in pun.items():
            tokenized_sents.append(word_tokenize(" ".join(word.values())))

    for sentence in tokenized_sents:
        sentence = [w.lower() for w in sentence if not w in stop_words]
        trigrams.append(list(ngrams(sentence, 3)))

    return trigrams
