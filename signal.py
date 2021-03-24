import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.linear_model import TheilSenRegressor

from helpers import load_fundamental, unify_idx, rolling_sum, unstack, end_of_the_month, shift_daily, load_daily, load_price


def load_signal(method="OLS", lagged=False) -> pd.DataFrame:
    
    ACO = load_fundamental("other_current_assets")
    AO = load_fundamental("other_non_current_assets")
    AP = load_fundamental("trade_payables")
    AT = load_fundamental("total_assets")
    CEQ1 = load_fundamental("total_equity")
    CEQ2 = load_fundamental("preferred_stock")
    CHE1 = load_fundamental("cash_and_cash_equivalent")
    CHE2 = load_fundamental("current_financial_assets")
    DLTT = load_fundamental("non_current_liabilities")
    DO = load_fundamental("discontinued_operation_income")
    DV = load_fundamental("dividend_paid")
    IB = load_fundamental("ongoing_operating_income")
    ICAPT = load_fundamental("invested_capital")
    LCO = load_fundamental("other_current_liabilities")
    LO = load_fundamental("other_non_current_liabilities")
    LT = load_fundamental("total_liabilities")
    NI = load_fundamental("net_income")
    NOPI = load_fundamental("non_operating_income")
    PI = load_fundamental("income")
    PPENT = load_fundamental("property_plant_and_equipment")
    PSTK = load_fundamental("preferred_stock")
    SALE = load_fundamental("sales")
    SEQ = load_fundamental("total_equity")
    TXT = load_fundamental("income_taxes_expenses")

    ACO, AO, AP, AT, CEQ1, CEQ2, CHE1, CHE2, DLTT, DO, DV, IB, ICAPT, LCO, LO, LT, NI, NOPI, PI, PPENT, PSTK, SALE, SEQ, TXT = unify_idx(ACO, AO, AP, AT, CEQ1, CEQ2, CHE1, CHE2, DLTT, DO, DV, IB, ICAPT, LCO, LO, LT, NI, NOPI, PI, PPENT, PSTK, SALE, SEQ, TXT)

    CEQ = CEQ1 - CEQ2
    CHE = CHE1 + CHE2

    AO = AO.fillna(0)
    DV = DV.fillna(0)
    LO = LO.fillna(0)

    DO = rolling_sum(DO, window=4)
    DV = rolling_sum(DV, window=4)
    IB = rolling_sum(IB, window=4)
    NI = rolling_sum(NI, window=4)
    NOPI = rolling_sum(NOPI, window=4)
    PI = rolling_sum(PI, window=4)
    SALE = rolling_sum(SALE, window=4)
    TXT = rolling_sum(TXT, window=4)

    ACO = unstack(ACO)
    AO = unstack(AO)
    AP = unstack(AP)
    AT = unstack(AT)
    CEQ = unstack(CEQ)
    CHE = unstack(CHE)
    DLTT = unstack(DLTT)
    DO = unstack(DO)
    DV = unstack(DV)
    IB = unstack(IB)
    ICAPT = unstack(ICAPT)
    LCO = unstack(LCO)
    LO = unstack(LO)
    LT = unstack(LT)
    NI = unstack(NI)
    NOPI = unstack(NOPI)
    PI = unstack(PI)
    PPENT = unstack(PPENT)
    PSTK = unstack(PSTK)
    SALE = unstack(SALE)
    SEQ = unstack(SEQ)
    TXT = unstack(TXT)
    
    if lagged:
        ACO = shift_daily(ACO, month=6)
        AO = shift_daily(AO, month=6)
        AP = shift_daily(AP, month=6)
        AT = shift_daily(AT, month=6)
        CEQ = shift_daily(CEQ, month=6)
        CHE = shift_daily(CHE, month=6)
        DLTT = shift_daily(DLTT, month=6)
        DO = shift_daily(DO, month=6)
        DV = shift_daily(DV, month=6)
        IB = shift_daily(IB, month=6)
        ICAPT = shift_daily(ICAPT, month=6)
        LCO = shift_daily(LCO, month=6)
        LO = shift_daily(LO, month=6)
        LT = shift_daily(LT, month=6)
        NI = shift_daily(NI, month=6)
        NOPI = shift_daily(NOPI, month=6)
        PI = shift_daily(PI, month=6)
        PPENT = shift_daily(PPENT, month=6)
        PSTK = shift_daily(PSTK, month=6)
        SALE = shift_daily(SALE, month=6)
        SEQ = shift_daily(SEQ, month=6)
        TXT = shift_daily(TXT, month=6)

    mkt_cap = load_daily("mkt_cap", market=["KSE", "KOSDAQ"])
    price_close = load_price(method="raw", market=["KSE", "KOSDAQ"])
    
    end_of_the_month_ = end_of_the_month()
    
    signal = pd.DataFrame()

    for date in tqdm(end_of_the_month_):
        temp = pd.DataFrame()

        for i, df in enumerate([ACO, AO, AP, CEQ, CHE, DLTT, DO, DV, IB, ICAPT, LCO, LO, LT, NI, NOPI, PI, PPENT, PSTK, SEQ, TXT]):
            temp = pd.concat([temp, pd.DataFrame({i: df.loc[date,:].dropna()})], axis=1)

        temp.columns = ["ACO", "AO", "AP", "CEQ", "CHE", "DLTT", "DO", "DV", "IB", "ICAPT", "LCO", "LO", "LT", "NI", "NOPI", "PI", "PPENT", "PSTK", "SEQ", "TXT"]    
        temp = pd.concat([temp, pd.DataFrame({"mkt_cap": mkt_cap.loc[date,:].dropna()})], axis=1).dropna()

        if not temp.empty and temp.shape[0] > temp.shape[1] - 1:
            if method == "OLS":
                temp = sm.add_constant(temp)
                model = sm.OLS(temp["mkt_cap"], temp[list(set(temp.columns) - set(["mkt_cap"]))]).fit()
            elif method == "TS":
                model = TheilSenRegressor(random_state=0, fit_intercept=True).fit(X=temp[list(set(temp.columns) - set(["mkt_cap"]))], y=temp["mkt_cap"])
                
            pred = model.predict(temp[list(set(temp.columns) - set(["mkt_cap"]))])

            signal = signal.append(pd.DataFrame({date: (pred - temp["mkt_cap"]) / temp["mkt_cap"]}).T)
            
    signal.to_csv("./data/signal_%s_%s.csv" % (method, "lagged" if lagged else ""), index=True)
    
    return signal

