from ERModel.models.document import Document as Doc
import pickle
import re


class Reader:
    def __init__(self) -> None:
        pass


    def read_dataset(path=None, delimiter=';'):
        res = []
        try:
            with open(path) as file:
                for line in file:
                    splitted = re.split(delimiter, line)
                    res.append(Doc(splitted[0], splitted[1].strip()))
        except Exception as e:
            print(f"Exception : {e}")

        return res
    

def read_pickled_obj(path):
    with open(path, 'rb') as file:
            res = pickle.load(file)
    return res