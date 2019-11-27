import xml.etree.ElementTree as ET
import os
import time
import string
from nltk.corpus import stopwords

DATA_DIR    = os.path.join(".", "semeval2017_task7", "data", "test")
RESULTS_DIR = os.path.join(".", "results")

def get_puns(subtask=2, h="homographic"):
    """
    Create a dictionary nested in a dictionary that contains all puns
    of one subtask, either homographic or heterographic.
    
    To get a specific word in a (possible) pun from the nested dictionary:
    puns["hom_2250"]["hom_2250_1"]

    return also the subtask ID, e.g. "subtask2-homographic"
    """
    filename = "subtask" + str(subtask) + "-" + h + "-test" + ".xml"
    root = ET.parse(os.path.join(DATA_DIR, filename)).getroot()

    puns = {}
    for pun in root:
        puns[pun.attrib['id']] = {}
        for word in pun:
            puns[pun.attrib['id']][word.attrib['id']] = word.text

    taskID = root.attrib['id']
    return puns, taskID


def get_all_puns():
    """
    Create a three-fold nested dictionary to get all the puns of all subtasks.

    To get a specific word in a (possible) pun from the nested dictionary:
    all_puns["subtask2-homographic"]["hom_2250"]["hom_2250_4"]
    """
    all_puns = {}
    for task in range(1,4):
        puns, taskid = get_puns(subtask=task, h="homographic")
        all_puns[taskid] = puns
        puns, taskid = get_puns(subtask=task, h="heterographic")
        all_puns[taskid] = puns
    return all_puns


def write_results(results, filename="results", timestamp=True):
    """
    Write results into a text file from a list of tuples.
    """
    if timestamp:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename += timestr
    filename += ".txt"

    with open(os.path.join(RESULTS_DIR, filename), "w") as f:
        for r in results:
            f.write(r[0] + " " + r[1] + "\n")


def remove_punctuation(puns):
    """
    Remove punctuation characters from a puns dictionary.
    """
    new_puns = {}
    for punID, pun in puns.items():
        new_puns[punID] = {}
        for wordID, word in pun.items():
            if word not in string.punctuation:
                new_puns[punID][wordID] = word

    return new_puns


def remove_stopwords(puns):
    """
    Remove stop words from a puns dictionary.
    """
    stopWords = set(stopwords.words('english'))
    new_puns = {}
    for punID, pun in puns.items():
        new_puns[punID] = {}
        for wordID, word in pun.items():
            if word not in stopWords:
                new_puns[punID][wordID] = word

    return new_puns
