import numpy as np

def cosine_similarity(vec1, vec2):
    return vector_dot_product(vec1, vec2) / (vector_magnitude(vec1) * vector_magnitude(vec2))


def vector_dot_product(vec1, vec2):
    return np.dot(vec1, vec2)

def vector_magnitude(vec):
    return np.sqrt(vec.dot(vec))