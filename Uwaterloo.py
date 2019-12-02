import process_data
import pickle

with open('tf-idf.pickle', 'rb') as handle:
    tf_idf = pickle.load(handle)
with open('df.pickle', 'rb') as handle:
    df = pickle.load(handle)

def select_tom_swifty(puns, results):
    for punID, pun in puns.items():
        if ('said', 'VBD') in set(pun.values()):
            for wordID, word in pun.items():
                if word[1] == 'RB' and word[0][-2:] == 'ly':
                    print(pun.values(), word[0])
                    results[punID] = wordID


# def score_tf_idf(puns):
#     for punID, pun in puns.items():
puns, asd = process_data.get_puns()
puns = process_data.only_content_words(process_data.add_pos_tags(process_data.remove_punctuation(puns) ))
keyerrors = 0

for punID, pun in puns.items():
    for wordID, word in pun.items():
        try:
            df[word[0].lower()]
        except KeyError:
            keyerrors += 1
            try:
                print(word[0].lower())
            except UnicodeEncodeError:
                print("unicode error")
            

print(keyerrors)

        