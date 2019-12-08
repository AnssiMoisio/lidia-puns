import process_data
import pickle


def get_homophone_dict():
    """
    Get a dictionary of similarly sounding words from "homophones.txt"
    """
    with open('homophones.txt', 'r') as f:
        l = f.read().splitlines()

    l = [line.replace(',', ' ').split() for line in l]

    homophone_dict = {}
    for line in l:
        for word in line:
            if word not in homophone_dict:
                homophone_dict[word] = set(line)
            else:
                homophone_dict[word] = homophone_dict[word].union(set(line))
    
    for word, set_of_homophones in homophone_dict.items():
        if word in set_of_homophones:
            set_of_homophones.remove(word)

    return homophone_dict


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


