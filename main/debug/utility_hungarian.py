if __name__ == '__main__':
    from utility import Hungarian
    import numpy as np

    hg = Hungarian()
    # cost matrix
    cost = [[0, 0, 3, 2], [3, 5, 1, 0], [5, 9, 0, 3], [2, 2, 0, 1]]
    optim = hg.compute(cost)
    print(optim)

