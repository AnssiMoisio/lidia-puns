import process_data
import string
import operator
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
    largest_word_number = 0
    for wordID, word in pun.items():
        if word['word_number'] > largest_word_number:
            largest_word_number = word['word_number']
            lastID = wordID

    return lastID


def select_last_word(puns):
    """
    Select the last word of the pun in the pun location task.
    Return a dictionary: {punID: lastwordID}
    """
    results = {}
    for punID, pun in puns.items():
        results[punID] = get_last_word(pun)
    return results


def select_last_word_exclude_most_common(puns):
    """
    Select the last word of the pun in the pun location task.
    Exclude most common words in the Brown Corpus
    Return a dictionary: {punID: lastwordID}
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

'''
puns, taskID = process_data.get_puns()
# puns = process_data.truncate_puns(puns, keep=30)
process_data.lowercase_caps_lock_words(puns)
process_data.add_pos_tags(puns)
process_data.lowercase(puns)
puns = process_data.only_content_words(puns)
puns = process_data.remove_stopwords(puns)
process_data.add_word_numbers(puns)
results = select_last_word_exclude_most_common(puns)
process_data.write_results(results, filename=taskID + "-baseline-exclude-" + str(n_exclude) + "-most-common", timestamp=False)
'''