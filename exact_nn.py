import numpy as np
from perf import perf_timed
from glove import glove


def exact_nearest_neighbors(token, n=100):
    """ nth nearest neighbors as array
        with indices of nearest neighbors"""
    glove_matrix, idx_to_token, token_to_idx = glove()
    token_idx = token_to_idx[token]
    token_vect = glove_matrix[token_idx]

    dotted = np.dot(glove_matrix, token_vect)
    normed = np.linalg.norm(glove_matrix, axis=1)
    nn = np.divide(dotted,normed)
    top_n = np.argpartition(-nn, n)[:n]
    return top_n, nn[top_n]

def print_nearest_neighbors(token):
    nn, scores = exact_nearest_neighbors(token, n=30)
    _, idx_to_token, _ = glove()
    for idx, score in zip(nn, scores):
        print(score, idx_to_token[idx])


if __name__ == "__main__":
    from sys import argv
    from download import download
    download(['https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/embeddings/glove/glove.6B.300d.npz'])
    tokens=argv[1:]
    for token in tokens:
        print("==========================")
        print("%s nn:" % token)
        with perf_timed():
            print_nearest_neighbors(token)
