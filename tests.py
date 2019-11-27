import unittest
import process_data

class TestDataProcessing(unittest.TestCase):
    def test_get_puns(self):
        puns, taskID = process_data.get_puns()
        self.assertEqual(puns["hom_2250"], {'hom_2250_1': 'A', 'hom_2250_2': 'student', 'hom_2250_3': 'limped', 'hom_2250_4': 'into', 'hom_2250_5': 'class', 'hom_2250_6': 'with', 'hom_2250_7': 'a', 'hom_2250_8': 'lame', 'hom_2250_9': 'excuse', 'hom_2250_10': '.'} )

    def test_remove_punctuation(self):
        puns, taskID = process_data.get_puns()
        p = process_data.remove_punctuation(puns)
        self.assertEqual(p["hom_2250"], {'hom_2250_1': 'A', 'hom_2250_2': 'student', 'hom_2250_3': 'limped', 'hom_2250_4': 'into', 'hom_2250_5': 'class', 'hom_2250_6': 'with', 'hom_2250_7': 'a', 'hom_2250_8': 'lame', 'hom_2250_9': 'excuse'})

    def test_remove_stopwords(self):
        puns, taskID = process_data.get_puns()
        p = process_data.remove_stopwords(puns)
        self.assertEqual(p["hom_2250"], {'hom_2250_1': 'A', 'hom_2250_2': 'student', 'hom_2250_3': 'limped', 'hom_2250_5': 'class','hom_2250_8': 'lame', 'hom_2250_9': 'excuse', 'hom_2250_10': '.'} )

if __name__ == '__main__':
    unittest.main()