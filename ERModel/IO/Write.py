import pickle
class Writer:
    """
        writes an object into a file in binary using pickle class.

        inputs
            path: the path to a file into which the object is to be written.

            obj: the object to be pickled into the file using pickle class.

        returns
            None
    """
    def write_pickled_obj(path, obj):
        try:
            with open(path, 'wb') as file:
                pickle.dump(obj, file)
            return True
        except:
            return False