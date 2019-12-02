import process_data

def select_last_word(puns):
    """
    Select the last word of the pun in the pun location task.
    Return a dictionary: {punID: lastwordID}
    """
    results = {}
    for punID, pun in puns.items():
        for wordID, word in pun.items():
            pass # iterate all words just to get last wordID
        results[punID] = wordID

    return results

