import process_data

def select_last_word(puns):
    """
    Select the last word of the pun in the pun location task.
    Return a list of tuples: (punID, lastwordID)
    """
    results = []
    for punID, pun in puns.items():
        for wordID, word in pun.items():
            pass # iterate all words just to get last wordID
        results.append( (punID, wordID) )

    return results


## create baselines

puns, taskID = process_data.get_puns()
"""
puns = process_data.remove_punctuation(puns)
results = select_last_word(puns)
process_data.write_results(results, filename=taskID + "-last-word-baseline-no-punctuation", timestamp=False)

puns = process_data.remove_stopwords(puns)
results = select_last_word(puns)
process_data.write_results(results, filename=taskID + "-last-word-baseline-no-punctuation-no-stopwords", timestamp=False)
"""
puns = process_data.only_content_words(process_data.add_pos_tags(puns))
# results = select_last_word(puns)
# process_data.write_results(results, filename=taskID + "-last-word-baseline-only-content-words", timestamp=True)
print(puns["hom_631"])
# puns = process_data.remove_stopwords(puns)
# puns = process_data.only_content_words(process_data.add_pos_tags(puns))
# print(puns['hom_1'])
results = select_last_word(puns)
process_data.write_results(results, filename=taskID + "-last-word-baseline-no-stopwords-only-content-words", timestamp=True)
