import process_data
import pickle


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
