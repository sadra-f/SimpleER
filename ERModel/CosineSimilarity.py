import numpy as np

def cosine_similarity(vec1, vec2):
    """
        calculates and returns the cosine similarity between vec1 and vec2 using numpy methods.
    """
    return vector_dot_product(vec1, vec2) / (vector_magnitude(vec1) * vector_magnitude(vec2))


def vector_dot_product(vec1, vec2):
    """
        calculates and returns the dot product of vec1 and vec2 using numpy methods.
    """
    return np.dot(vec1, vec2)

def vector_magnitude(vec):
    """
        calculates and returns the magnitude of vec.
    """
    return np.sqrt(vec.dot(vec))