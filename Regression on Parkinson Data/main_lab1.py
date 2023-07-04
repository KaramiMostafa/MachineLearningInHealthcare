import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libraries import *

#%% Data Prepration :
x = pd.read_csv("parkinsons_updrs.csv")  #
x.info()  # the panda framework display an easier to read matrix


#%% Covariance matrix of the features
xnorm = (x - x.mean()) / x.std()
cc = xnorm.cov()


#%% Plotting :
features = [
    "age",
    "sex",
    "motor_UPDRS",
    "total_UPDRS",
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

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(np.abs(cc))
ax.set_yticks(np.arange(len(features)))
ax.set_yticklabels(features)
fig.colorbar(im)
plt.title("Covariance matrix of the features")
plt.savefig("Covariance matrix of the features", dpi=200)


#%% subject and time are dropped but the ones with higher covariance are kept :
features_to_drop = [
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
x1 = x.copy(deep=True)
x1 = x1.drop(features_to_drop, axis=1)
xnorm1 = (x1 - x1.mean()) / x1.std()
c1 = xnorm1.cov()

f = ["subject#", "test_time"]
x2 = x.copy(deep=True)
x2 = x2.drop(f, axis=1)


#%% division in training, test, valid (the training data is 50% of the whole data set,
# validation 25% and test data set 25%) :
np.random.seed(11)

data = x2.sample(frac=1).reset_index(drop=True)

Np = data.shape[0]
Nf = data.shape[1]

data_train = data[0 : int(Np / 2)]
data_val = data[int(Np / 2) : int(Np * 0.75)]
data_test = data[int(Np * 0.75) : Np]


mean = np.mean(data_train.values, 0)
std = np.std(data_train.values, 0)


data_train_nonorm = data_train.values
data_train_norm = (data_train.values - mean) / std

data_val_nonorm = data_val.values
data_val_norm = (data_val.values - mean) / std

data_test_nonorm = data_test.values
data_test_norm = (data_test.values - mean) / std


F = 3  # the feature wanted to choose to be the regressand -------> the Total UPDRS

# removing the total_updrs column
y_train = data_train_norm[:, F]
y_trr = np.reshape(y_train, (2937, 1))
x_train = np.delete(data_train_norm, F, 1)

y_test = data_test_norm[:, F]
y_txt = np.reshape(y_test, (1469, 1))
x_test = np.delete(data_test_norm, F, 1)

y_val = data_val_norm[:, F]
y_valid = np.reshape(y_val, (1469, 1))
x_val = np.delete(data_val_norm, F, 1)


#%% Initialization

logx = 0
logy = 1

lls_stat = np.zeros((3, 2), dtype=float)
conj_stat = np.zeros((3, 2), dtype=float)
sgwa_stat = np.zeros((3, 2), dtype=float)
ridge_stat = np.zeros((3, 2), dtype=float)

R2 = np.zeros((4, 1), dtype=float)


mat = np.zeros((3, 4), dtype=float)


# matrixes to compare mse values on normalized results
mse_train = np.zeros((4, 1), dtype=float)
mse_test = np.zeros((4, 1), dtype=float)
mse_val = np.zeros((4, 1), dtype=float)


#%% LLS

m = lls(y_train, x_train, y_test, x_test, y_val, x_val, mean[F], std[F])
mse_train[0], mse_test[0] = m.run()
mat[:, 0] = m.MSE_un()
lls_stat = m.stat_u()
m.plot_w("optimum weight vector - LLS")
m.comparison(
    "y VS y\u0302 training - LLS",
    "y\u0302 training",
    "y training",
    y_train,
    x_train,
    mean[F],
    std[F],
)  # sto estraendo la media e la varianza della colonna total updrs
m.comparison(
    "y VS y\u0302 test - LLS", "y\u0302 test", "y test", y_test, x_test, mean[F], std[F]
)  # sto estraendo la media e la varianza della colonna total updrs
m.hist(y_train, x_train, y_test, x_test, mean[F], std[F], " - LLS")
R2[0] = m.determ_coeff()


#%% CONJUGATE

c = CONJ(y_trr, x_train, y_txt, x_test, y_valid, x_val, mean[F], std[F])
mse_train[1], mse_val[1], mse_test[1] = c.run()
mat[:, 1] = c.MSE_un()

c.plot_w("optimum weight vector - CGA")
c.comparison(
    "y VS y\u0302 training - Conjugate",
    "y\u0302 training",
    "y training",
    y_trr,
    x_train,
    mean[F],
    std[F],
)
c.comparison(
    "y VS y\u0302 test - Conjugate",
    "y\u0302 test",
    "y test",
    y_test,
    x_test,
    mean[F],
    std[F],
)
c.hist_plus_val(
    y_trr, x_train, y_txt, x_test, y_valid, x_val, mean[F], std[F], " - CGA"
)
c.plot_err("square error - Conjugate", logy, logx)
conj_stat = c.stat_u()
R2[1] = c.determ_coeff()


#%% STOCHASTIC GRADIENT WITH ADAM

s = SGDA(y_trr, x_train, y_txt, x_test, y_valid, x_val, mean[F], std[F])
mse_train[2], mse_val[2], mse_test[2] = s.run(
    y_valid, x_val, y_txt, x_test, 0.001, 0.99, 0.999, 20000, 1e-8
)
mat[:, 2] = s.MSE_un()

s.plot_w("optimum weight vector - Stochastic gradient with Adam")
s.plot_err("square error - Stochastic gradient with Adam", logy, logx)
s.comparison(
    "y VS y\u0302 test - SGD with ADAM",
    "y\u0302 test",
    "y test",
    y_test,
    x_test,
    mean[F],
    std[F],
)
s.hist_plus_val(
    y_trr, x_train, y_txt, x_test, y_valid, x_val, mean[F], std[F], " - SGwA"
)
sgwa_stat = s.stat_u()
R2[2] = s.determ_coeff()


#%% RIDGE

r = SolveRidge(y_trr, x_train, y_txt, x_test, y_valid, x_val, mean[F], std[F])
mse_train[3], mse_val[3], mse_test[3] = r.run()
mat[:, 3] = r.MSE_un()
r.plot_w("optimum weight vector (Ridge)")
r.comparison(
    "y VS y\u0302 test - Ridge",
    "y\u0302 test",
    "y test",
    y_test,
    x_test,
    mean[F],
    std[F],
)
r.hist_plus_val(
    y_trr, x_train, y_txt, x_test, y_valid, x_val, mean[F], std[F], " - Ridge"
)
R2[3] = r.determ_coeff()
ridge_stat = r.stat_u()
