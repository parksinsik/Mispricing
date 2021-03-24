import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta


def load_fundamental(item: str) -> pd.DataFrame:
    
    data = pd.read_csv("./data/%s.csv" % item, parse_dates=["pit"])
    
    for i in ["1Q", "2Q", "3Q", "4Q"]:
        filt = data[data["fiscal"].str[4:6] == i]
        filt.loc[:,"year"] = filt.loc[:,"fiscal"].str[0:4].astype(int) + 1
        filt.set_index(["short_codes", "year"], inplace=True)
        
        if i == "1Q":
            filt_index = set(filt.index)
        else:
            filt_index = filt_index.intersection(set(filt.index))
            
    filt_index = sorted(filt_index)

    data.loc[:,"year"] = data.loc[:,"fiscal"].str[0:4].astype(int)
    data.set_index(["short_codes", "year"], inplace=True)

    data = data.loc[sorted(set(data.index).intersection(set(filt_index))),:]
    data.index.names = ["short_codes", "year"]
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    data.set_index(["short_codes", "fiscal", "pit"], inplace=True)
    data.drop(columns=["year"], inplace=True)

    data = data.reorder_levels(["short_codes", "fiscal", "pit"])
    data.index = data.index.set_levels(data.index.levels[0].astype(int), level=0)
    
    return data.sort_index()


def unify_idx(*args):    
    
    idx = pd.MultiIndex.from_tuples(sorted(set().union(*map(lambda x: x.index, args))), names=["short_codes", "fiscal", "pit"])
    
    result = []
    
    for arg in tqdm(args):
        arg = arg.reindex(idx)
        arg.reset_index(inplace=True)
        arg["value"] = arg.groupby(["short_codes", "fiscal"])["value"].transform(lambda x: x.ffill()).values
        arg.drop_duplicates(subset=["short_codes", "pit"], keep="last", inplace=True)
        arg.set_index(["short_codes", "fiscal", "pit"], inplace=True)
        
        result.append(arg)
    
    return result


def _shift_fundamental(data, periods: int) -> pd.DataFrame:

    assert periods > 0

    data_original = data.copy()
    data_original.reset_index(inplace=True)
    data_original.loc[:,"order"] = data_original.loc[:,"fiscal"].str[2:4].astype(int) * 4 + data_original.loc[:,"fiscal"].str[4].astype(int)

    data_shifted = data_original.copy()
    data_shifted["order"] += periods

    data_original.set_index(["short_codes", "order"], inplace=True)
    data_shifted.set_index(["short_codes", "order"], inplace=True)

    result = pd.merge(data_original, data_shifted, how="left", left_index=True, right_index=True)
    result = result[["fiscal_x", "pit_x", "value_y"]]
    result.rename(columns={"fiscal_x": "fiscal", "pit_x": "pit", "value_y": "value"}, inplace=True)
    result.reset_index(inplace=True)
    result.drop(columns=["order"], inplace=True)

    return result.drop_duplicates(subset=["short_codes", "pit"], keep="last").set_index(["short_codes", "fiscal", "pit"])


def rolling_sum(data, window: int) -> pd.DataFrame:
    
    assert window > 1
    
    for i in range(window - 1):
        data += _shift_fundamental(data, i+1)
    
    return data


def unstack(data) -> pd.DataFrame:
    return data[np.isfinite(data)].reset_index("fiscal", drop=True)["value"].unstack(level=0).reindex(pd.date_range(datetime(2000, 1, 1), datetime(2020, 12, 31))).ffill(limit=400)


def end_of_the_month() -> pd.Series:    
    data = _load_daily("mkt_cap")
    data = pd.DataFrame({"dates": data.index})
    data["ym"] = data["dates"].apply(lambda x: datetime.strftime(x, "%Y%m"))
    data = data.groupby("ym").last()
    data.reset_index(drop=True, inplace=True)
    return data["dates"][:-1]


def shift_daily(data, month=None) -> pd.DataFrame:

    assert month is not None
    
    start, end = data.index[0], data.index[-1]

    all_dates = pd.date_range(start, end)
    trading_dates = data.index
    end_of_the_month_ = end_of_the_month()

    data.index.name = "pit"

    data_shifted = data.copy()
    data_shifted.index.name = "pit"
    data_shifted.index = pd.Series(data_shifted.index).apply(lambda x: x + relativedelta(months=month))
    data_shifted.reset_index(inplace=True)
    data_shifted.drop_duplicates(subset=["pit"], keep="last", inplace=True)
    data_shifted.set_index("pit", inplace=True)
    data_shifted = data_shifted.fillna(np.inf).reindex(all_dates).ffill().reindex(trading_dates).replace(np.inf, np.nan)
    
    result = data_shifted.append(data.reindex(end_of_the_month_).shift(month))
    result.index.name = "pit"
    result.reset_index(inplace=True)
    result.drop_duplicates(subset=["pit"], keep="last", inplace=True)
    result.set_index("pit", inplace=True)

    return result.sort_index()


def load_daily(item: str, market=["KSE", "KOSDAQ"]) -> pd.DataFrame:
    if market is None:
        return _load_daily(item)
    else:
        result = _load_daily(item) * _load_daily("market").isin(market).replace(np.nan, False).astype(int)
        return result.replace(0, np.nan)
    

def _load_daily(item: str) -> pd.DataFrame:
    data = pd.read_csv("./data/%s.csv" % item, parse_dates=[0])
    data.set_index(data.columns[0], inplace=True)
    data.index.name = None
    data.columns = data.columns.astype(int)
    return data


def load_price(method="raw", market=["KSE", "KOSDAQ"]) -> pd.DataFrame:
    
    data = load_daily("price_close", market=market)
    data = data.pct_change(periods=1, fill_method=None)
    data = data.fillna(((data.isna() & data.shift().notna()) * -0.99).replace(0, np.nan))
    
    if method == "raw":
        return data
    elif method == "industry_adj":
        return _industry_adj(data)
    else:
        raise

    
def _industry_adj(data) -> pd.DataFrame:

    industry = _load_daily("industry")

    result = pd.DataFrame()
    
    for date in tqdm(data.index):
        temp = pd.concat([data.loc[date, :], industry.loc[date, :]], axis=1)
        temp.columns = "ret", "industry"

        avg = temp.groupby("industry", squeeze=True, group_keys=False).mean()
        
        temp = pd.merge(temp, avg, how="inner", left_on="industry", right_index=True)
        temp = temp["ret_x"] - temp["ret_y"]
        
        result = result.append(pd.DataFrame({date: temp}).T)

    return result
