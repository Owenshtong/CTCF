### Perform the KMV estimation on extended Merton's model ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fun
import quote_wrds
from def_cls import DefaultFirm
import quote_wrds as quote

# risk-free rate
zcy = pd.read_csv("zcy.csv", index_col=0, parse_dates=True)/100
zcy = zcy.interpolate(method="time")


# company input
companies = pd.read_excel('sp500_default.xlsx',
                          sheet_name="ListofDefault",
                          dtype=str
                          )
companies = companies.dropna()
companies = companies.astype({
    "bond_fv_mill": float,
    "gvkey": str,
    "permno": str
    })

# companies.t_start = (pd.to_datetime(companies.t_start) + pd.DateOffset(months=-2)).dt.strftime('%Y-%m-%d')
# companies.t_end = (pd.to_datetime(companies.t_end) + pd.DateOffset(months=-2)).dt.strftime('%Y-%m-%d')



# Check if there's debt info available
l=[]
for i in range(len(companies)):
    # print(i)
    firm = companies.iloc[i,:]
    l.append(quote.check_bond_existence(firm.t_start,firm.t_end,firm.tic))

companies["Available"] =  l
companies = companies[companies.Available]
companies = companies[~companies.tic.isin(["WIN", "DNR", "PQ"])]


# Get a list of DefaultFrim objects
l_data = []
for i in range(len(companies)):
    firm = companies.iloc[i,:]
    if firm.tic in ["WIN", "DNR", "PQ"]: # Firm PQ does not have enough yield data
        continue
    else:
        print(firm.tic)
        data = DefaultFirm(
            firm.tic,
            firm.t_start,
            firm.t_end,
            firm.bond_maturity,
            firm.permno,
            firm.gvkey,
            firm.bond_fv_mill * 1e6,
            zcy
        )
        l_data.append(data)


companies.iloc[:, [0,1,2,3,9,10,11]].to_csv("aaa")
# get lambda
l_sigma = []
l_vo = []
l_lambda = []
l_lambda_inno = []
x0 = 0.05
meth = "nelder-mead"

for i in range(30):
    f_profile = l_data[i]
    # f_profile.data = f_profile.data[f_profile.data.yld <= 1]
    sigma, v0, lambda_0, y_mod, acc = fun.lambda_finder(
                      f_profile.data.Mrk_cap,
                      f_profile.senior_debt_nominal_value,
                      f_profile.data.K,
                      f_profile.data.rf,
                      f_profile.senior_debt_maturity_date,
                      f_profile.data.yld,
                      x0,
                      meth
                      )
    l_sigma.append(sigma)
    l_vo.append(v0)
    l_lambda.append(lambda_0[0])
    l_lambda_inno.append(acc[0])

col_name = ["sigma", "v0", "lambda", "Lambda_last_innov"]
res_no_filtration = pd.DataFrame([l_sigma,l_vo,l_lambda,l_lambda_inno]).T
res_no_filtration.columns = col_name




#------------------ Experiments --------------------#
f = l_data[0]
# f.data = f.data[f.data.yld <=1]


x0 = 0.2
meth = "L-BFGS-B"
sigma, v0, lambda_0, y_mod, acc = fun.lambda_finder(
    f.data.Mrk_cap,
    f.senior_debt_nominal_value,
    f.data.K,
    f.data.rf,
    f.senior_debt_maturity_date,
    f.data.yld,
    x0,
    meth
)
# plt.plot(f.data.yld)
# plt.plot(y_mod)
# plt.show()


### yield value v.s. lambda: How sensitive?
daytime = np.vectorize(
    pd.to_datetime
)
dates = daytime(f.data.index)
_N_days = np.array([(pd.to_datetime(f.senior_debt_maturity_date) - i).days for i in dates])
T_yobs = _N_days / 365
lambda_range = np.linspace(0,0.05,50)
l_yld = []
l_sse = []
for lamb in lambda_range:
    d_lamb = fun.D(v0, sigma, lamb, f.senior_debt_nominal_value, f.data.rf, T_yobs)
    yld_lamb = fun.yld_mod(d_lamb, f.senior_debt_nominal_value, T_yobs)
    l_sse.append(sum(yld_lamb - f.data.yld)**2)
    l_yld.append(yld_lamb)

# yld curve
for i in range(50):
    plt.plot(l_yld[i])
# plt.plot(f.data.yld)
plt.show()

# lambda and sse
plt.scatter(lambda_range, l_sse)
plt.show()


l_s = []
v_range = np.linspace(min(f.data.Mrk_cap), max(f.data.Mrk_cap), 5)
for v in v_range:
    l_s.append(fun.S(v, sigma, lambda_0, f.data.K[-1], f.data.rf[-1], T = 1))

plt.plot(v_range,l_s)
plt.show()


l_s = []
for t in np.linspace(0.00001,1,20):
    l = []
    for v in np.linspace(2.4e7, max(f.data.K), 100):
        l.append(fun.S(v, sigma, lambda_0, f.data.K[-1], f.data.rf[-1], T = t)[0])
    l_s.append(l)
    plt.plot(np.linspace(2.2e8, 2.5e8, 100),l)
plt.show()


# toy example
fig,ax = plt.subplots(figsize = (10,8))
l_s = []
for t in [1e-10, 0.01, 0.1, 0.5, 1,2,3,5,8]:
    l = []
    for v in np.linspace(40,100,100):
        l.append(fun.S(v, 0.1, np.array([0.1]), 80, 0.02, T = t)[0])
    l_s.append(l)
    ax.plot(np.linspace(40,100,100),l, label = "T = " + str(t))
ax.plot([40, 80],[4, 8], color = "blue", label = "pay-off at maturity")
ax.plot([80, 100],[0, 20],color = "blue")
plt.title("Extened Merton model payoff over time")
plt.legend()
plt.gcf()
plt.savefig("payoff.pdf")
plt.show()


### Debt vs lambda:
from scipy.stats import norm

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
    # print(d1)
    # print(d2)
    return M * np.exp(-r * T) * norm.cdf(d2) + (1 - lamb) * v * norm.cdf(-d1)

l_D = []
lamb_range = np.linspace(0,1,100)
for d in lamb_range:
    l_D.append(
        D(v0, sigma, d, f.senior_debt_nominal_value, f.data.rf[-1], 5)
    )
plt.plot( lamb_range, l_D)
plt.show()





