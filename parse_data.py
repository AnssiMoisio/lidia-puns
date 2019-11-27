import xml.etree.ElementTree as ET
import os

DATA_DIR = os.path.join(".", "semeval2017_task7", "data", "test")

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
