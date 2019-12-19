import xml.etree.ElementTree as ET
import os
import time
import string
from nltk import pos_tag
from nltk.corpus import stopwords
import collections

DATA_DIR    = os.path.join(".", "semeval2017_task7", "data", "test")
RESULTS_DIR = os.path.join(".", "results")

def get_puns(subtask=2, h="homographic", truncate=None):
    """
    Create a dictionary nested in a dictionary that contains all puns
    of one subtask, either homographic or heterographic.
    
    To get a specific word in a (possible) pun from the nested dictionary:
    puns["hom_2250"]["hom_2250_1"]["token"]

    return also the subtask ID, e.g. "subtask2-homographic"
    """
    filename = "subtask" + str(subtask) + "-" + h + "-test" + ".xml"
    root = ET.parse(os.path.join(DATA_DIR, filename)).getroot()

    puns = {}
    for t, pun in enumerate(root):
        if t == truncate: break
        punID = pun.attrib['id']
        puns[punID] = collections.OrderedDict()
        for word in pun:
            wordID = word.attrib['id']
            puns[punID][wordID]           = {}
            puns[punID][wordID]['token']  = word.text

            # add the number of word e.g. "hom_345_13" -> word["word_number"]: 13 
            _counter = 0
            for char_ind in range(len(wordID)):
                if wordID[char_ind] == "_":
                    _counter += 1
                if _counter == 2:
                    puns[punID][wordID]['word_number'] = int(wordID[char_ind + 1:])
                    break

            # whether the word is the pun word or not
            if subtask == 3:
                if int(word.attrib['senses']) > 1:
                    puns[punID][wordID]['ispun'] = True
                else:
                    puns[punID][wordID]['ispun'] = False

    taskID = root.attrib['id']
    return puns, taskID

def get_pun_tokens(pun, exclude={}, return_pos_tags=False):
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

def only_content_words(puns):
    """
    Keep only nouns, verbs, adverbs and adjectives in the puns dictionary.
    Assumes puns dictionary includes POS tags.
    """
    content_tags = ['FW', 'JJ', 'NN', 'RB', 'VB'] # first 2 letters of POS tag
    new_puns = {}
    for punID, pun in puns.items():
        new_puns[punID] = {}
        for wordID, word in pun.items():
            if word['pos'][:2] in content_tags:
                new_puns[punID][wordID] = word

    return new_puns
