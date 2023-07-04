import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib.pyplot import figure

###################################################


class SolvMinProbl:
    def __init__(self, y, A, y_test, A_test, y_val, A_val, mean, var):

        self.matr = A
        self.Np = A.shape[0]
        self.Nf = A.shape[1]

        self.matr_test = A_test
        self.matr_val = A_val

        self.mean = mean
        self.var = var

        self.vect = y
        self.vect_test = y_test
        self.vect_val = y_val

        self.sol = np.zeros((self.Nf, 1), dtype=float)
        self.err = 0
        self.mse = 0
        self.stat = 0

    def MSE_un(self):

        self.mse = np.zeros((3, 1), dtype=float)

        y_train_estimated = self.var * np.dot(self.matr, self.sol) + self.mean
        y_train = self.var * self.vect + self.mean

        y_validation = self.var * self.vect_val + self.mean
        y_val_estimated = self.var * np.dot(self.matr_val, self.sol) + self.mean

        ytest = self.var * self.vect_test + self.mean
        y_test_estimated = self.var * np.dot(self.matr_test, self.sol) + self.mean

        self.mse[0] = (
            np.linalg.norm(y_train - y_train_estimated) ** 2
        ) / self.matr.shape[0]
        self.mse[1] = (
            np.linalg.norm(y_validation - y_val_estimated) ** 2
        ) / self.matr_val.shape[0]
        self.mse[2] = (
            np.linalg.norm(ytest - y_test_estimated) ** 2
        ) / self.matr_test.shape[0]

        return self.mse[0], self.mse[1], self.mse[2]

    #%% Statistical Properties:
    def stat_u(self):
        self.stat = np.zeros((3, 2), dtype=float)
        y_train_estimated = self.var * np.dot(self.matr, self.sol) + self.mean
        y_train = self.var * self.vect + self.mean
        errtr = y_train - y_train_estimated
        self.stat[0][0] = errtr.mean()
        self.stat[0][1] = errtr.std()

        y_validation = self.var * self.vect_val + self.mean
        y_val_estimated = self.var * np.dot(self.matr_val, self.sol) + self.mean
        errva = y_validation - y_val_estimated
        self.stat[1][0] = errva.mean()
        self.stat[1][1] = errva.std()

        ytest = self.var * self.vect_test + self.mean
        y_test_estimated = self.var * np.dot(self.matr_test, self.sol) + self.mean
        errte = ytest - y_test_estimated
        self.stat[2][0] = errte.mean()
        self.stat[2][1] = errte.std()

        return self.stat

    #%% Coefficient Determination:
    def determ_coeff(self):
        ytest = self.var * self.vect_test + self.mean
        y_test_estimated = self.var * np.dot(self.matr_test, self.sol) + self.mean

        r2 = r2_score(ytest, y_test_estimated)

        return r2

    #%% Plotting Functions :
    def plot_w(self, title):
        # I want to plot the weight vector for all the considered features
        plt.figure()
        plt.figure(figsize=(7, 4))
        w = self.sol
        n = np.arange(self.Nf)
        plt.plot(n, w)
        features = [
            "age",
            "sex",
            "motor_UPDRS",
            "Jitter(%)",
            "Jitter(Abs)",
            "Jitter:RAP",
            "Jitter:PPQ5",
            "Jitter:DDP",
            "Shimmer",
            "Shimmer(dB)",
            "Shimmer:APQ3",
            "Shimmer:APQ5",
            "Shimmer:APQ11",
            "Shimmer:DDA",
            "NHR",
            "HNR",
            "RPDE",
            "DFA",
            "PPE",
        ]
        plt.xticks(n, features, rotation="vertical")
        # plt.ylim(-50, 50)
        plt.xticks(np.arange(0, 19, 1))
        plt.ylabel("weight vector")
        plt.title(title)
        plt.grid()

        plt.savefig(title + ".png", dpi=200, bbox_inches="tight")
        plt.show()

    def comparison(self, title, labelx, labely, y, A, mean, var):

        plt.figure()
        w = self.sol
        y_est = (np.dot(A, w) * var) + mean
        y = (y * var) + mean  # un-normalized y
        plt.plot(np.linspace(0, 60), np.linspace(0, 60), "red")
        plt.scatter(y, y_est, s=5, color="forestgreen")  # forestgreen color for test
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.title(title)
        plt.grid()
        plt.savefig(title + ".png", dpi=200, bbox_inches="tight")
        plt.show()

    def hist(self, y, A, y_test, A_test, mean, var, title):
        plt.figure()
        plt.figure(figsize=(7, 4))
        w = self.sol
        y_tr = (y * var) + mean  # un-normalized y
        y_est_tr = (np.dot(A, w) * var) + mean
        y_test = (y_test * var) + mean  # un-normalized y
        y_est_test = (np.dot(A_test, w) * var) + mean
        plt.hist(
            y_tr - y_est_tr, bins=100, alpha=0.7, color="darkviolet", label="training"
        )
        plt.hist(
            y_test - y_est_test, bins=100, alpha=0.7, color="forestgreen", label="test"
        )
        plt.xlabel("Estimation Error")
        plt.ylabel("frequency")
        plt.title("Estimation Error Histogram" + title)
        plt.legend(loc="best")
        plt.savefig(title, dpi=200)
        plt.show()

    def hist_plus_val(self, y, A, y_test, A_test, y_val, A_val, mean, var, title):
        plt.figure()
        plt.figure(figsize=(7, 4))
        w = self.sol
        y_tr = (y * var) + mean  # un-normalized y
        y_est_tr = (np.dot(A, w) * var) + mean

        y_test = (y_test * var) + mean  # un-normalized y
        y_est_test = (np.dot(A_test, w) * var) + mean

        y_val = (y_val * var) + mean  # un-normalized y
        y_est_val = (np.dot(A_val, w) * var) + mean

        plt.hist(
            y_tr - y_est_tr, bins=100, alpha=0.7, label="training", color="darkviolet"
        )
        plt.hist(
            y_test - y_est_test, bins=100, alpha=0.7, label="test", color="forestgreen"
        )
        plt.hist(
            y_val - y_est_val, bins=100, alpha=0.7, label="validation", color="gold"
        )

        plt.xlabel("Estimation Error")
        plt.ylabel("frequency")
        plt.title(
            "Histogram for the Estimation Error for Training, Validation & Test Data"
            + title
        )
        plt.legend(loc="best")
        plt.savefig(title, dpi=200)
        plt.show()

    #%% Parameters :
    def plot_err(self, title, logy=1, logx=0):

        err = self.err
        plt.figure()
        if (logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 1], label="train", color="darkviolet")
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1], label="train", color="darkviolet")
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1], label="train", color="darkviolet")
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 1], label="train", color="darkviolet")
        if (logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 2], label="val", color="orange", alpha=0.6)
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 2], label="val", color="orange", alpha=0.6)
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 2], label="val", color="orange", alpha=0.6)
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 2], label="val", color="orange", alpha=0.6)
        plt.xlabel("n")
        plt.ylabel("e(n)")
        plt.legend()
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.savefig(title + ".png", dpi=200)
        plt.show()


#%% LLS method :
class lls(SolvMinProbl):
    def run(self):

        A = self.matr
        A_test = self.matr_test

        y = self.vect
        y_test = self.vect_test

        w = np.random.rand(self.Nf, 1)
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)
        self.sol = w

        self.err = np.zeros((1, 4), dtype=float)

        # computation for mse_train/test/val
        self.err[0, 1] = np.linalg.norm(y - np.dot(A, w)) ** 2 / A.shape[0]
        self.err[0, 3] = (
            np.linalg.norm(y_test - np.dot(A_test, w)) ** 2 / A_test.shape[0]
        )
        return self.err[0, 1], self.err[0, 3]


#%% Conjugate method :
class CONJ(SolvMinProbl):
    def run(self):

        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
        A_test = self.matr_test
        y_test = self.vect_test
        w = np.zeros((self.Nf, 1), dtype=float)
        self.err = np.zeros((self.Nf, 4), dtype=float)
        it = 0
        Q = 2 * np.dot(A.T, A)
        b = 2 * np.dot(A.T, y)
        g = -b
        d = -g
        for it in range(self.Nf):

            a = -1 * np.dot(d.T, g) / np.dot(np.dot(d.T, Q), d)
            w = w + a * d
            g = g + a * np.dot(Q, d)
            beta = np.dot(np.dot(g.T, Q), d) / np.dot(np.dot(d.T, Q), d)
            d = -1 * g + beta * d
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(y - np.dot(A, w)) ** 2 / A.shape[0]
            self.err[it, 2] = (
                np.linalg.norm(y_val - np.dot(A_val, w)) ** 2 / A_val.shape[0]
            )
            self.err[it, 3] = (
                np.linalg.norm(y_test - np.dot(A_test, w)) ** 2 / A_test.shape[0]
            )
        self.sol = w
        return self.err[-1, 1], self.err[-1, 2], self.err[-1, 3]


#%% Stochastic Gradient Decent with Adam method :
class SGDA(SolvMinProbl):
    def run(self, y_val, A_val, y_test, A_test, gamma, Beta1, Beta2, it, epsilon):

        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)
        self.MSE = np.zeros((it, 4), dtype=float)
        self.err = np.zeros((it, 4), dtype=float)

        mu1 = 0
        mu2 = 0
        mu1_hat = 0
        mu2_hat = 0

        stop = 0

        for t in range(it):
            if stop < 50:
                grad = 2 * np.dot(A.T, (np.dot(A, w) - y))
                mu1 = mu1 * Beta1 + (1 - Beta1) * grad
                mu2 = mu2 * Beta2 + (1 - Beta2) * pow(grad, 2)

                mu1_hat = mu1 / (1 - pow(Beta1, (t + 1)))
                mu2_hat = mu2 / (1 - pow(Beta2, (t + 1)))

                w = w - gamma * mu1_hat / (np.sqrt(mu2_hat) + epsilon)

                self.MSE[t, 0] = t
                self.MSE[t, 1] = np.linalg.norm(y - np.dot(A, w)) ** 2 / A.shape[0]
                self.MSE[t, 2] = (
                    np.linalg.norm(y_val - np.dot(A_val, w)) ** 2 / A_val.shape[0]
                )
                self.MSE[t, 3] = (
                    np.linalg.norm(y_test - np.dot(A_test, w)) ** 2 / A_test.shape[0]
                )

                if self.MSE[t, 2] > self.MSE[t - 1, 2]:
                    stop += 1
                else:
                    stop = 0
            else:
                print("The number of Nit is :", t)
                break

        self.sol = w
        self.err = self.MSE[0:t]

        return self.err[-1, 1], self.err[-1, 2], self.err[-1, 3]


#%% Ridge regression method :
class SolveRidge(SolvMinProbl):
    def run(self):
        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
        A_test = self.matr_test
        y_test = self.vect_test
        I = np.eye(self.Nf)
        lamb = 300
        self.err = np.zeros((lamb, 4), dtype=float)
        for it in range(lamb):
            w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A) + it * I), A.T), y)
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(y - np.dot(A, w)) ** 2 / A.shape[0]
            self.err[it, 2] = (
                np.linalg.norm(y_val - np.dot(A_val, w)) ** 2 / A_val.shape[0]
            )
            # print(self.err[it, 2])
            self.err[it, 3] = (
                np.linalg.norm(y_test - np.dot(A_test, w)) ** 2 / A_test.shape[0]
            )
        best_lamb = np.argmin(self.err[:, 2])
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A) + best_lamb * I), A.T), y)
        self.sol = w
        err = self.err
        print("The best lambda is :", best_lamb)

        # plotting the MSE with respect to lambda
        plt.figure(figsize=(7, 4))
        plt.semilogy(err[:, 0], err[:, 1], label="train")
        plt.semilogy(err[:, 0], err[:, 2], label="val")
        plt.xlabel("lambda")
        plt.ylabel("error")
        plt.legend()
        plt.title("MSE")
        plt.grid()
        plt.savefig("MSE", dpi=200)
        plt.show()

        return self.err[-1, 1], self.err[-1, 2], self.err[-1, 3]
