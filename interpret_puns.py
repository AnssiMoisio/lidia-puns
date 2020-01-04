import common
import string
import time
from nltk.corpus import wordnet as wn, stopwords, brown
from nltk.util import ngrams

def create_word_pairs_file(puns, filename="pun_pairs", add_ngrams=True, n=3):
    """
    write a file with all the context words paired with the pun word
    e.g.
    
    #hom_2
    Wal saving
    Mart saving
    isn saving
    t saving
    the saving
    only saving
    saving place

    optionally also n-grams with the pun word
    
    """
    all_word_pairs =[]
    for punID, pun in puns.items():
        word_pairs = []
        word_pairs.append([punID])
        for wordID, word in pun.items():
            if word['ispun']:
                punword = word
        for wordID, word in pun.items():
            word_pair = []
            if not word['ispun']:
                if word['word_number'] < punword['word_number']:
                    word_pair.append(word['token'])
                    word_pair.append(punword['token'])
                else:
                    word_pair.append(punword['token'])
                    word_pair.append(word['token'])
                word_pairs.append(word_pair)

        if add_ngrams:
            n_grams = list(ngrams(common.get_pun_tokens(pun), n))
            for gram in n_grams:
                if punword['token'] in gram:
                    word_pairs.append(gram)
                
        all_word_pairs.append(word_pairs)

    with open(filename, "w") as f:
        for punpairs in all_word_pairs:
            for pair in range(len(punpairs)):
                if pair == 0:
                    f.write("#")
                    f.write("".join(punpairs[pair]))
                    f.write("\n")
                else:
                    f.write(" ".join(punpairs[pair]) + "\n")
            f.write("\n")

def get_pun_token(pun):
    """
    Return a set the contains the pun word and its lemma
    """
    for wordID, word in pun.items():
        try:
            if word['ispun']:
                return wordID, set([word['token'], wn.morphy(word['token'])])
        except:
            pass

def parse_wsd_file(puns, filename="wsd_output.txt"):
    """
    add the sensekeys list in the puns dict as the value of pun['sensekeys']
    """
    with open(filename, "r") as f:
        output = f.read().splitlines()

    for line in output:
        if len(line) > 0:
            if line[0] == "#":
                punID = line[1:-1]
                puns[punID]['sensekeys'] = []
            else:
                tokens = line.split(" ")
                tokens = [item.split("|") for item in tokens if item != '']
                for line in tokens:
                    try:
                        if len(line) > 1 and line[0] in get_pun_token(puns[punID])[1]:
                            puns[punID]['sensekeys'].append(line[1])
                    except TypeError as err:
                        print(err, "something is wrong here")

def create_results(puns):
    """
    create the results file from the puns dict
    """
    results = {}
    for punID, pun in puns.items():
        try:
            puntokenid, punwords = get_pun_token(pun)
        except:
            continue
        maxcount = 0
        secondcount = 0
        bestsense = ""
        secondsense = ""
        try:
            for sense in set(pun['sensekeys']):
                n = pun['sensekeys'].count(sense)
                if n > maxcount:
                    secondsense = bestsense
                    secondcount = maxcount
                    maxcount = n
                    bestsense = sense
                elif n >= secondcount:
                    secondcount = n
                    secondsense = sense
            if bestsense != "" and secondsense != "":
                results[puntokenid] = bestsense + " " + secondsense
        except KeyError:
            pass
    return results


puns, taskID = common.get_puns(subtask=3, truncate=None)
common.lowercase_caps_lock_words(puns)
common.add_pos_tags(puns)
common.lowercase(puns)
puns = common.remove_punctuation(puns)
# puns = common.only_content_words(puns)
# puns = common.remove_stopwords(puns)

# create_word_pairs_file(puns, filename="wsd_input.txt", add_ngrams=True, n=5)

parse_wsd_file(puns, filename='wsd_output.txt')
res = create_results(puns)
common.write_results(res, filename="5grams2", timestamp=False)

