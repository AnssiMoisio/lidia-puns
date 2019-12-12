import process_data
import Uwaterloo
import string
import numpy as np
import copy
from nltk import FreqDist
from nltk.corpus import wordnet as wn, stopwords, brown

stopWords = set(stopwords.words('english'))

# retrieve the n_exclude most common words from the Brown Corpus, exclude stopwords and punctuation
n_exclude = 200
words = brown.words()
filtered_words = [w.lower() for w in words if w.lower() not in stopWords and w not in string.punctuation and w not in "''``'--"]
word_freq = FreqDist(w for w in filtered_words)
most_common_words = set()
for wordtuple in list(word_freq.most_common(n_exclude)):
    most_common_words.add(wordtuple[0])


def get_last_word(pun):
    """
    return the wordID (e.g. "hom_2_4") with the largest word number from this pun
    """
    lastID = None
    largest_word_number = 0
    for wordID, word in pun.items():
        if word['word_number'] > largest_word_number:
            largest_word_number = word['word_number']
            lastID = wordID

    if lastID is not None:
        return lastID
    else:
        print("get last word error")


def select_last_word(puns):
    """
    Select the last word of the pun in the pun location task.
    Return a results dictionary: {punID: lastwordID}
    """
    results = {}
    for punID, pun in puns.items():
        results[punID] = get_last_word(pun)
    return results


def select_last_word_exclude_most_common(puns):
    """
    Select the last word of the pun in the pun location task.
    Exclude most common words in the Brown Corpus
    Return a results dictionary: {punID: lastwordID}
    """
    results = {}
    for punID, pun in puns.items():
        while True:
            lastID = get_last_word(pun)
            if pun[lastID]['token'] not in most_common_words:
                break
            else:
                del pun[lastID]
        results[punID] = lastID

    return results


def select_least_common_of_last_n_words(pun, n):
    """
    Select the least common word of the n last words of pun.
    use the Brown Corpus
    Return the word ID.
    """
    ID = ""
    new_pun = copy.deepcopy(pun)
    last_n_words = [""] * n
    word_freqs = np.zeros((n))
    for i in range(n):
        last_n_words[i] = get_last_word(new_pun)
        try:
            del new_pun[last_n_words[i]]
        except KeyError:
            break
        w = pun[last_n_words[i]]['token']
        # w = wn.morphy(w) if wn.morphy(w) is not None else w
        word_freqs[i] = word_freq[w]

    # least_common
    ID = last_n_words[np.argmin(word_freqs)]

    return ID

def select_word_with_lowest_freq(pun):
    """
    Select the least common word.
    use the Brown Corpus
    Return the word ID.
    """
    min_freq = np.inf
    minID = ""
    for wordID, word in pun.items():
        try:
            f = word_freq[word['token']]
        except KeyError:
            f = 0
        if f < min_freq:
            min_freq = f
            minID = wordID

    # print(minID, pun[minID]['token'],pun[minID]['pos'], min_freq)
    return minID


puns, taskID = process_data.get_puns(h="heterographic", truncate=None)
process_data.lowercase_caps_lock_words(puns)
process_data.add_pos_tags(puns)
process_data.lowercase(puns)
puns = process_data.only_content_words(puns)
puns = process_data.remove_stopwords(puns)
process_data.add_word_numbers(puns)

results = {}
for punID, pun in puns.items():
    wordID = select_word_with_lowest_freq(pun)
    # if pun[wordID]['token'] == 'Tom' or pun[wordID]['word_number'] == 1:
    #     wordID = select_least_common_of_last_n_words(pun, 2)
    results[punID] = wordID

process_data.write_results(results, filename=taskID + "-test", timestamp=False)
