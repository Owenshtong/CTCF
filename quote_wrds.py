import numpy as np
import pandas as pd
import wrds
from datetime import datetime
from dateutil.relativedelta import relativedelta


conn = wrds.Connection(wrds_username='owen_tong')


# Define SQL query to retrieve daily price and shares outstanding
def quote_equity_CRSP(s0, sT, permno):
    query = "SELECT date, permno, prc, shrout " +\
            "FROM crsp.dsf " +\
            "WHERE permno = " + permno +\
            " AND date BETWEEN " + str("'" + s0 +"'" + " AND " + "'" + sT +"'") +\
            " ORDER BY date"
    S = conn.raw_sql(query)
    S["shrout"] = S["shrout"] * 1000
    S["Mrk_cap"] = S["prc"] * S["shrout"]

    S.index = pd.to_datetime(S.date)
    S = S.drop(["date", "permno"], axis = 1)
    return S


def quote_debt_compstat(s0, sT, gvkey):
    # Shift the starting time for 4 months
    s0_quater = datetime.strptime(s0, "%Y-%m-%d")
    s0_quater = s0_quater - relativedelta(months=4)
    s0_quater = s0_quater.strftime("%Y-%m-%d")

    sT_quater = datetime.strptime(sT, "%Y-%m-%d")
    sT_quater = sT_quater + relativedelta(months=4)
    sT_quater = sT_quater.strftime("%Y-%m-%d")

    query = (
        "SELECT "
        "c.gvkey, "
        "c.datadate, "
        "c.fyearq, "
        "c.fqtr, "
        "c.dlcq, "
        "c.dlttq, "
        "n.tic "
        "FROM "
        "comp.fundq AS c "
        "LEFT JOIN "
        "comp.names AS n "
        "ON "
        "c.gvkey = n.gvkey "
        "WHERE "
        "c.gvkey = '" + gvkey + "' "
        "AND c.datadate BETWEEN '" + s0_quater + "' AND '" + sT_quater + "' "
        "ORDER BY "
        "c.datadate;"
    )
    D = conn.raw_sql(query)
    D["dlcq"] = D["dlcq"] * 1e6
    D["dlttq"] = D["dlttq"] * 1e6


    D.index = pd.to_datetime(D.datadate)
    D = D.drop("datadate", axis = 1)

    return D[["dlcq", "dlttq"]]

def quote_yield_TRACE(s0, sT, sM, ticker):
    query = (
        "SELECT "
        "b.company_symbol,"
        "b.date,"
        "b.bond_sym_id, "
        "b.maturity "
        "FROM "
        "wrdsapps.bondret AS b "  # Query from the bondret dataset
        "WHERE "
        "b.company_symbol = '" + ticker + "' "  # Filter by firm ticker (AMC)
        "AND b.t_date BETWEEN '" + s0 + "' AND '" + sT + "' "
        "AND B.maturity = '" + sM + "'"
        "ORDER BY "
        "b.date;"
    )
    bond_id_data = conn.raw_sql(query)

    # Get longest unsecured debt id
    bond_id = bond_id_data.bond_sym_id.unique()[0]

    query = (
        "SELECT "
        "t.TRD_EXCTN_DT, "
        "t.YLD_PT, "
        "t.bond_sym_id "
        "FROM "
        "trace.trace AS t "  # Assuming this is the correct table in the TRACE database
        "WHERE "
        "t.bond_sym_id = '" + bond_id + "' "  # Filter by the given bond ID
        "AND t.trd_exctn_dt BETWEEN '" + s0 + "' AND '" + sT + "' "
        "ORDER BY "
        "t.trd_exctn_dt;"
    )
    y_obs = conn.raw_sql(query)
    y_obs["yld_pt"] = y_obs["yld_pt"]/100
    y_obs = y_obs[[ "trd_exctn_dt",     "yld_pt"]]

    # drop na
    y_obs = y_obs.dropna()

    # daily average of yield
    y_obs = y_obs.groupby("trd_exctn_dt").mean()

    # Keep those index matches the equity
    y_obs.index = pd.to_datetime(y_obs.index)

    return y_obs


def check_bond_existence(s0, sT, ticker):
    query = (
        "SELECT "
        "b.company_symbol,"
        "b.date,"
        "b.bond_sym_id, "
        "b.maturity "
        "FROM "
        "wrdsapps.bondret AS b "  # Query from the bondret dataset
        "WHERE "
        "b.company_symbol = '" + ticker + "' "  # Filter by firm ticker (AMC)
        "AND b.t_date BETWEEN '" + s0 + "' AND '" + sT + "' "
        "ORDER BY "
        "b.date;"
    )
    bond_id_data = conn.raw_sql(query)


    return not bond_id_data.empty
