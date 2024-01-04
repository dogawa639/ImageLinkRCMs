if __name__ == "__main__":
    from utility import KalmanFilter
    import numpy as np
    import matplotlib.pyplot as plt

    dim_x = 3
    dim_z = 2

    x = np.random.randn(1, dim_x)
    f = np.random.randn(dim_x, dim_x)
    h = np.random.randn(dim_z, dim_x)
    z = np.dot(h, x.T).T + np.random.randn(1, dim_z)

    def q_fn(x):
        return np.eye(dim_x)

    def r_fn(z):
        return np.eye(dim_z)

    kf = KalmanFilter(f, h, q_fn, r_fn)
    kf.add_obj(z)
    times = 10
    x_true = np.zeros((times, dim_x))
    x_pred = np.zeros((times, dim_x))
    p = np.zeros((times, dim_x))
    for t in range(times):
        x = np.dot(f, x.T).T
        z = np.dot(h, x.T).T + np.random.randn(1, dim_z)

        kf.predict()
        kf.correct(z)

        x_true[t] = x[0]
        x_pred[t] = kf.xnn[0]
        p[t] = kf.pnn[0].diagonal()

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    plt.legend()
    ax2 = fig.add_subplot(2, 1, 2)
    plt.legend()
    for i in range(dim_x):
        ax1.plot(x_true[:, i], label=f"true_{i}")
        ax2.plot(x_pred[:, i], label=f"pred_{i}")
        #ax2.plot(x_pred[:, i] + p[:, i], ":")
        #ax2.plot(x_pred[:, i] - p[:, i], ":")
    plt.show()


