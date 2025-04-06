import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve
from functools import partial
from scipy.optimize import minimize

# Equity value
def S(v, sigma, lamb, Kt, rt, T = 1):
    """
    :param T: maturity
    :param lamb: the recovery rate for shareholders
    :param v: firm or asset value
    :param sigma: vol of v
    :param Kt: series. total debt = .5 long + short
    :param rt: risk-free rate
    :return: The equity value at time 0
    """
    d1 = (np.log(v / Kt) + (rt + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return v * norm.cdf(d1) - Kt * np.exp(-1 * rt * T) * norm.cdf(d2) + lamb * v * norm.cdf(-d1)



def S_inv(S_obs, sigma, lamb, Kt, rt):
    """
    Same in put as S except the S value to be inversed
    :return: asset value at t = 0, v0
    """
    fun = lambda v: S(v, sigma, lamb, Kt, rt) - S_obs
    v0 = fsolve(
        fun,
        x0 = S_obs + Kt
    )
    return v0

# Debt value
def D(v, sigma, lamb, M, r, T):
    """
     The debt value at time 0
     :param T: maturity
     :param lamb: the recovery rate for shareholders
     :param v: firm or asset value at time 0
     :param sigma: vol of v
     :param M: The longest senior unsecured debt
     :param r: risk-free rate
     :return: Equity value
     """
    d1 = (np.log(v / M) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return M * np.exp(-r * T) * norm.cdf(d2) + (1 - lamb) * v * norm.cdf(-d1)

def log_return(S):
    """
    :param S: Time series of stock price
    :return: log-return
    """
    St = S
    St1 = S.shift(1).dropna() # shift forward by 1 period
    return np.log(St / St1).dropna()


# KMV Moody's method to imply the v0 and sigma with a given lambda
def KMV(lamb, S, K, r, tol = 1e-4):
    """
    :param lamb: lambda share recovered by shareholders
    :param K: total  debt .5 long + short
    :param r: risk-free rate
    :param S: time series of equity prices
    :param tol: Stopping criteria
    :return: vol and v0
    """

    # Vectorize the S_inv function
    # S_inv_vect = np.vectorize(S_inv, excluded = [ "sigma", "lamb", "K", "r", "T"])

    # Initial guess
    acc = 1  # The absolute diff from last sigma
    logr = log_return(S)
    sigma = np.sqrt(np.var(logr))
    n = 0 # number of accumulation
    n_max = 20
    while (acc > tol) & (n <= n_max):
        n += 1
        # v_inv = S_inv_vect(S, sigma = sigma, lamb = lamb, K = K, r = r)
        v_inv = [S_inv(S[i], sigma, lamb,  K[i], r[i])[0] for i in range(len(S))]
        v_inv = pd.Series(v_inv, index = S.index)
        v_logr = log_return(v_inv)
        acc = abs(np.sqrt(np.var(v_logr)) - sigma)
        sigma = np.sqrt(np.var(v_logr))
        vn = S_inv(S[-1], sigma,lamb,K[-1],r[-1])
        print("KMV: volatility improvement " + str(acc))
    return sigma, vn[0]


def argmin_lamb(v, sigma, M, maturity_date, r, y_obs, x0, meth):
    """
    :param r: risk-free rate
    :param maturity_date: Str, bond maturity date
    :param v:  Asset value
    :param sigma: vol
    :param M: the longest senior unsecured debt nominal value
    :param y_obs: Observed yield of the longest senior unsecured
    :return:
    """
    # Count the date to maturity
    daytime = np.vectorize(
        pd.to_datetime
    )
    dates = daytime(y_obs.index)
    _N_days = np.array([(pd.to_datetime(maturity_date) - i).days for i in dates])
    T_yobs = _N_days  / 365

    # Debt fun
    f_D_vect = lambda lamb : D(v,sigma, lamb, M, r, T_yobs)
    f_y_model = lambda lamb: -1/T_yobs * np.log(f_D_vect(lamb) / M)


    def sse(lamb):
        print(f_D_vect(lamb))
        print(f_y_model(lamb))
        s = sum((f_y_model(lamb) - y_obs)**2)
        print(s)
        return s

    # Minimize ssr
    opt_result = minimize(sse,
                          x0=x0,
                          method=meth,
                          bounds=[(0, 1)],
                          )
    opt_lambda = opt_result["x"]

    return opt_lambda, f_y_model(opt_lambda)

# Looking for the fixed point of recovery rate
def lambda_finder(S, M, K, r, maturity_date, y_obs, x0,meth, tol = 1e-3):
    lambda_0 = .5
    acc = 1
    n_max = 15
    n=0
    while (acc >= tol) & (n <= n_max):
        n += 1
        lambda_old = lambda_0
        sigma, v0 = KMV(lambda_0, S, K, r)
        lambda_0, y_mod = argmin_lamb(v0, sigma, M, maturity_date, r, y_obs,x0,meth)
        acc = abs(lambda_old - lambda_0)
        print(sigma, v0, lambda_0, acc)
    return  sigma, v0, lambda_0, y_mod, acc


### Helper ###
# Interpolation
def df_interpolate(s, unkonwn_dates):
    '''
    :param s: series with index type datetime
    :param know_dates: series with know value and dates
    :param unkonwn_dates: series of dates with values to be interpolated
    :return:
    '''
    # get unique index
    missing_df = pd.DataFrame(index=pd.to_datetime(unkonwn_dates))
    full_df = pd.concat([s, missing_df]).sort_index()

    # Perform time-based interpolation
    full_df = full_df.interpolate(method="time")

    return full_df.loc[unkonwn_dates]


def sse(lamb, v,sigma,M, maturity_date, r, y_obs):
    """
    :param r: risk-free rate
    :param maturity_date: Str, bond maturity date
    :param v:  Asset value
    :param sigma: vol
    :param M: the longest senior unsecured debt nominal value
    :param y_obs: Observed yield of the longest senior unsecured
    :return:
    """
    # Count the date to maturity
    daytime = np.vectorize(
        pd.to_datetime
    )
    dates = daytime(y_obs.index)
    _N_days = np.array([(pd.to_datetime(maturity_date) - i).days for i in dates])
    T_yobs = _N_days  / 365

    # Debt fun
    f_D_vect = lambda lamb : D(v,sigma, lamb, M, r, T_yobs)
    f_y_model = lambda lamb: -1/T_yobs * np.log(f_D_vect(lamb) / M)


    def sse(lamb):
        s = sum((f_y_model(lamb) - y_obs)**2)
        print(s)
        return s

    return sse(lamb)

def yld_mod(D, M, T):
    return (-1 / T) * np.log(D / M)

