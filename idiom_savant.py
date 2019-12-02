import process_data



def get_senses_of_word(word):
    """
    return the senses of word from wordnet
    """


def word_vector(word):
    """
    return the word embedding of a word
    """


def cosine_similarity(w1, w2):
    """
    Return the similarity of two word vectors
    """


def word_similarity_with_context(word, context):
    """
    Calculate similarity of the senses with every context word
    """
    word_embed = word_vector(word)
    sim_values = {}
    for context_word in context:
        sim_value[] = cosine_similarity(word_vector(context_word), word_embed)



    return score


def baseline_homographic():
    """
    return the word with the two senses with
    highest similarities with context
    """ 


def n_gram_prob(ngram):
    """
    search the n-gram probability (or count) from google n-grams
    """
    return prob