from __future__ import annotations

import logging

import pandas as pd

from simtradelab.grid_screener.config import RunConfig
from simtradelab.ptrade.api import PtradeAPI
from simtradelab.ptrade.context import Context
from simtradelab.ptrade.object import Portfolio
from simtradelab.service.data_server import DataServer

_OHLCV_FIELDS = ["open", "high", "low", "close", "volume"]


def _init_ptrade_api(
    data_path: str | None,
    market: str,
    *,
    quiet: bool = False,
) -> PtradeAPI:
    """与 strategies/grid_mining/miner.py 中 init_api 相同写法，仅 screener 自用。"""
    if not quiet:
        print("正在加载数据: price...")

    data_server = DataServer(
        required_data={"price"},
        data_path=data_path,
        market=market,
    )
    portfolio = Portfolio(initial_capital=100000)
    context = Context(portfolio=portfolio)
    log = logging.getLogger("grid_screener")
    if quiet:
        log.setLevel(logging.WARNING)

    api = PtradeAPI(data_context=data_server, context=context, log=log)

    if not quiet:
        keys_list = list(data_server.benchmark_data.keys())  # type: ignore[union-attr]
        print("✓ API 初始化完成")
        print("✓ 可用基准(共 {0} 个): {1} ...".format(len(keys_list), ", ".join(keys_list[:10])))

    return api


def _latest_trade_date_from_api(api: PtradeAPI) -> pd.Timestamp:
    """未配置 as_of 时，用已加载行情中的最近交易日（优先基准）。"""
    bench = getattr(api.data_context, "benchmark_data", None) or {}
    for df in bench.values():
        if isinstance(df, pd.DataFrame) and not df.empty:
            return pd.Timestamp(df.index[-1]).normalize()
    stock_dict = getattr(api.data_context, "stock_data_dict", None)
    if stock_dict:
        first = next(iter(stock_dict.values()), None)
        if isinstance(first, pd.DataFrame) and not first.empty:
            return pd.Timestamp(first.index[-1]).normalize()
    return pd.Timestamp.now().normalize()


class ScreenerDataAPI:
    """通过 PtradeAPI.get_price 取行情（与 miner.py 相同，不直读 storage）。"""

    def __init__(self, cfg: RunConfig, *, quiet: bool = True) -> None:
        self.cfg = cfg
        self._api = _init_ptrade_api(cfg.data_path, cfg.market, quiet=quiet)
        if cfg.as_of:
            self._end = pd.Timestamp(cfg.as_of).normalize()
        else:
            self._end = _latest_trade_date_from_api(self._api)
        self._api.context.current_dt = self._end.to_pydatetime()

    def _end_date_str(self) -> str:
        return str(self._end.date())

    def list_symbols(self) -> list[str]:
        return sorted(self._api.get_Ashares(self._end_date_str()))

    def get_stock_name(self, symbol: str) -> str:
        name = self._api.get_stock_name(symbol)
        if not name:
            return ""
        return str(name).replace("\x00", "").strip()

    def load_ohlcv(self, symbol: str) -> pd.DataFrame:
        df = self._api.get_price(
            symbol,
            end_date=self._end_date_str(),
            count=self.cfg.params.window_trading_days,
            frequency="1d",
            fields=_OHLCV_FIELDS,
            fq=self.cfg.resolved_fq(),
        )
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=_OHLCV_FIELDS)
        return df
