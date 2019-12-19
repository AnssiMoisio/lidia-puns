import common
import string
import time
from nltk.corpus import wordnet as wn, stopwords, brown

def create_word_pairs_file(puns, filename="pun_pairs"):
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

# def divide_pun(puns, filename="divided_pun"):
#     all_pun_parts = []
#     for punID, pun in puns.items():
#         pun_parts = []
#         pun_parts.append([punID])
#         for wordID, word in pun.items():
#             if word['ispun']:
#                 punword = word
#         for wordID, word in pun.items():
#             punpart = []



#             if not word['ispun']:
#                 if word['word_number'] < punword['word_number']:
#                     word_pair.append(word['token'])
#                     word_pair.append(punword['token'])
#                 else:
#                     word_pair.append(punword['token'])
#                     word_pair.append(word['token'])
#                 word_pairs.append(word_pair)
                
#         all_word_pairs.append(word_pairs)

#     with open(filename, "w") as f:
#         for punpairs in all_word_pairs:
#             for pair in range(len(punpairs)):
#                 if pair == 0:
#                     f.write("#")
#                     f.write("".join(punpairs[pair]))
#                     f.write("\n")
#                 else:
#                     f.write(" ".join(punpairs[pair]) + "\n")
#             f.write("\n")

def get_pun_token(pun):
    """
    Return a set the contains the pun word and it's lemma
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
                        if len(line) > 1 and line[0] in get_pun_token(puns[punID]):
                            puns[punID]['sensekeys'].append(line[1])
                    except TypeError as err:
                        print(err, "somethisng is wrong here")


puns, taskID = common.get_puns(subtask=3, truncate=None)
common.lowercase_caps_lock_words(puns)
common.add_pos_tags(puns)
common.lowercase(puns)
puns = common.only_content_words(puns)
puns = common.remove_stopwords(puns)


# create_word_pairs_file(puns, filename="wsd_input")
parse_wsd_file(puns)

for punID, pun in puns.items():
    print(pun['sensekeys'])


def create_results(puns):
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
                elif n > secondcount:
                    secondcount = n
                    secondsense = sense
            if bestsense != "" and secondsense != "":
                results[puntokenid] = bestsense + " " + secondsense
        except KeyError:
            pass
    return results

res = create_results(puns)
common.write_results(res, filename=taskID, timestamp=True)

