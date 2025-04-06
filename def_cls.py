### A class object for default companies ###
import numpy as np
import pandas as pd
import quote_wrds as wrds_quote
from functools import reduce
import fun

class DefaultFirm:
    def __init__(self, ticker, s0, sT, sM, permno, gvkey, nominal, rt):
        '''
        :param ticker: firm's ticker
        :param s0: start date
        :param sT: end date
        :param sM: maturity date of the debt/bond of interest
        :param permno: CRSP identifer of a firm
        :param gvkey: Compustat firm identifier
        :param nominal: Face vlaue of the longest unsecured date, Can be obtained from sp500 report
        '''
        self.gvkey = gvkey
        self.ticker = ticker
        self.permno = permno

        self.start_date = s0
        self.end_date = sT
        self.senior_debt_maturity_date = sM
        self.senior_debt_nominal_value = nominal

        # Raw data
        self.equity = wrds_quote.quote_equity_CRSP(self.start_date, self.end_date,self.permno)
        self.debt = wrds_quote.quote_debt_compstat(self.start_date, self.end_date,self.gvkey)
        self.senior_bond_yield = wrds_quote.quote_yield_TRACE(self.start_date, self.end_date, self.senior_debt_maturity_date, self.ticker)

        # Interpolate: Interpolate debt
        self.debt = fun.df_interpolate(self.debt, self.senior_bond_yield.index).drop_duplicates()

        # merger together
        self.data = reduce(lambda left, right: pd.merge(left, right,  how="inner", right_index=True, left_index=True),
                [self.equity, self.debt, self.senior_bond_yield, rt])
        self.data["K"] = self.data.dlcq + .5 * self.data.dlttq
        self.data.columns = ['S', 'N_shares', 'Mrk_cap', 'sd', 'ld', 'yld', 'rf', "K"]










