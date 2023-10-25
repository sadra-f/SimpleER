import pickle


def write_pickled_obj(path, obj):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)