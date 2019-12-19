import common
from nltk.util import ngrams
import urllib
import requests

def create_homophone_dict():
    """
    Create a dictionary of similarly sounding words from "homophones.txt"
    """
    with open('homophones.txt', 'r') as f:
        homophones = f.read().splitlines()

    homophones = [line.replace(',', ' ').split() for line in homophones]

    homophone_dict = {}
    for line in homophones:
        for word in line:
            homophone_set = set(line).difference(set([word]))
            if word not in homophone_dict:
                homophone_dict[word] = homophone_set
            else:
                homophone_dict[word].update(homophone_set)

    return homophone_dict

def get_idiom_freq(idiom):
    """
    Retrieve the frequency of an idiom from Google n-gram database.
    details: https://phrasefinder.io/api
    """
    encoded_query = urllib.parse.quote(idiom)
    params = {'corpus': 'eng-us', 'query': encoded_query}
    params = '&'.join('{}={}'.format(name, value) for name, value in params.items())
    response = requests.get('https://api.phrasefinder.io/search?' + params)
    assert response.status_code == 200

    if response.json()['phrases']:
        return response.json()['phrases'][0]['mc']
    else:
        return 0

def select_tom_swifty(puns, results):
    """
    Use some heuristics to identify and select the pun word for Tom Swifty puns.

    probably not useful
    """
    for punID, pun in puns.items():
        if ('said', 'VBD') in set(pun.values()):
            for wordID, word in pun.items():
                if word[1] == 'RB' and word[0][-2:] == 'ly':
                    print(pun.values(), word[0])
                    results[punID] = wordID

def get_trigrams(puns):
    """
    Separate the context into trigrams
    Return a list of tuples for each pun: [(word1, word2, word3), (word2, word3, word4)]
    """
    trigrams = {}
    for punID, pun in puns.items():
        tokenized_sent = common.get_pun_tokens(pun)
        trigrams[punID] = list(ngrams(tokenized_sent, 3))

    return trigrams