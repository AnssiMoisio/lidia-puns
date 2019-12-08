import process_data
import pickle


def get_homonym_dict():
    """
    Get a dictionary of similarly sounding words from "homonyms.txt"
    """
    with open('homonyms.txt', 'r') as f:
        l = f.read().splitlines()

    l = [line.replace(',', ' ').split() for line in l]

    homonym_dict = {}
    for line in l:
        for word in line:
            if word not in homonym_dict:
                homonym_dict[word] = set(line)
            else:
                homonym_dict[word] = homonym_dict[word].union(set(line))
    
    for word, set_of_homonyms in homonym_dict.items():
        if word in set_of_homonyms:
            set_of_homonyms.remove(word)

    return homonym_dict


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


