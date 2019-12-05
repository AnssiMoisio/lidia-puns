import unittest
import process_data
import idiom_savant

class TestDataProcessing(unittest.TestCase):
    # def test_get_puns(self):
    #     puns, taskID = process_data.get_puns()
    #     self.assertEqual(puns["hom_2250"], {'hom_2250_1': 'A', 'hom_2250_2': 'student', 'hom_2250_3': 'limped', 'hom_2250_4': 'into', 'hom_2250_5': 'class', 'hom_2250_6': 'with', 'hom_2250_7': 'a', 'hom_2250_8': 'lame', 'hom_2250_9': 'excuse', 'hom_2250_10': '.'} )

    # def test_remove_punctuation(self):
    #     puns, taskID = process_data.get_puns()
    #     p = process_data.remove_punctuation(puns)
    #     self.assertEqual(p["hom_2250"], {'hom_2250_1': 'A', 'hom_2250_2': 'student', 'hom_2250_3': 'limped', 'hom_2250_4': 'into', 'hom_2250_5': 'class', 'hom_2250_6': 'with', 'hom_2250_7': 'a', 'hom_2250_8': 'lame', 'hom_2250_9': 'excuse'})

    # def test_remove_stopwords(self):
    #     puns, taskID = process_data.get_puns()
    #     p = process_data.remove_stopwords(puns)
    #     self.assertEqual(p["hom_2250"], {'hom_2250_1': 'A', 'hom_2250_2': 'student', 'hom_2250_3': 'limped', 'hom_2250_5': 'class','hom_2250_8': 'lame', 'hom_2250_9': 'excuse', 'hom_2250_10': '.'} )

    def test_lowercase(self):
        puns, taskID = process_data.get_puns()
        puns = process_data.truncate_puns(puns, keep=500)
        puns = process_data.add_pos_tags(puns)
        puns = process_data.remove_stopwords(puns)
        puns = process_data.only_content_words(puns)
        print(puns["hom_556"])
        print(puns["hom_631"])
        puns = process_data.lowercase(puns)
        print(puns["hom_556"])
        print(puns["hom_631"])


class TestIdiomSavant(unittest.TestCase):
    def test_get_gloss(self):
        senses = idiom_savant.get_senses_of_word("sting")
        print(idiom_savant.get_gloss_set(senses[0]))

    # def test_word_vector(self):
    #     idiom_savant.word_vector("grey")

if __name__ == '__main__':
    unittest.main()