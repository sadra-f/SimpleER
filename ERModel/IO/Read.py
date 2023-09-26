from ERModel.models.document import Document as Doc
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
    