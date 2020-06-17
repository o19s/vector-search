import numpy as np

class RandomProjections:

    """ A Naive random projections """

    def __init__(self, matrix, num_vects=2000):
        cols=matrix.shape[1]
        # Generate random vects, balance around 0
        self.projs=np.random.rand( cols, num_vects  )
        self.projs -= 0.5
        normed = np.linalg.norm(self.projs, axis=1)
        self.projs = np.divide(self.projs.transpose(), normed).transpose()

        # Create a hash for each row
        self.hashed=np.matmul(matrix, self.projs)
        self.hashed[self.hashed >= 0] = 1
        self.hashed[self.hashed <  0] = -1
        self.hashed=self.hashed.astype(int)

    def __call__(self, row, matrix, n=100):
        # should just do this once

        # Dot with row to get sim score

        nn = np.dot(self.hashed, self.hashed[row])

        # Rank top N rows
        top_n = np.argpartition(-nn, n)[:n]
        return top_n, nn[top_n]


def rand_proj_from_glove():
    from glove import glove
    glove_matrix, _, _ = glove()
    return RandomProjections(glove_matrix)



if __name__ == "__main__":
    from algo_test import eval_algo
    algo = rand_proj_from_glove()
    eval_algo(algo)
