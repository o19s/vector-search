import numpy as np

def token_to_idx(idx_to_token):
    lookup={}
    for idx, token in enumerate(idx_to_token):
        lookup[token]=idx
    return lookup

from functools import lru_cache
@lru_cache(maxsize=1)
def glove(path):
    with np.load(path) as f:
        return f['idx_to_vec'], f['idx_to_token'], token_to_idx(f['idx_to_token'])


def nearest_neighbors(token, n=100):
    glove_matrix, idx_to_token, token_to_idx = glove('glove.6B.50d.npz')
    token_idx = token_to_idx[token]
    token_vect = glove_matrix[token_idx]

    dotted = np.dot(glove_matrix, token_vect)
    normed = np.linalg.norm(glove_matrix, axis=1)
    nn = np.divide(dotted,normed)
    top_n = np.argpartition(-nn, n)[:n]
    return top_n

def print_nearest_neighbors(token):
    nn = nearest_neighbors(token, n=5)
    _, idx_to_token, _ = glove('glove.6B.50d.npz')
    for idx in nn:
        print(idx_to_token[idx])


from contextlib import contextmanager
@contextmanager
def perf_log():
    from time import perf_counter
    start = perf_counter()
    yield start
    stop = perf_counter()
    print("Took %s" % (stop-start))



if __name__ == "__main__":
    from sys import argv
    tokens=argv[1:]
    for token in tokens:
        print("==========================")
        print("%s nn:" % token)
        with perf_log():
            print_nearest_neighbors(token)

