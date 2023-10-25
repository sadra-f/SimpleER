from ERModel.models.document import Document as Doc
import pickle
import re


class Reader:
    def __init__(self) -> None:
        """
            reader class used to read the dataset txt files returning each document/sentence as Document class objects.
        """
        pass


    def read_dataset(path=None, delimiter=';'):
        """
        reads the dataset from txt file at path.

        inputs

            path: the path to dataset txt file.

            delimiter: the character seperating the text/sentence from the emotion/class that represents it.


        returns
            list of Document class where each element is a dataset document along its emotion.
        """
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
        """
        reads the binary pickled object at the path.

        inputs
            the path at which the pickled file exists.
        
        returns
            the objct read from the path.
        """
        with open(path, 'rb') as file:
                res = pickle.load(file)
        return res