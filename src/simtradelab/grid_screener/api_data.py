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


class ScreenerDataAPI:
    """通过 PtradeAPI.get_price 取行情（与 miner.py 相同，不直读 storage）。"""

    def __init__(self, cfg: RunConfig, *, quiet: bool = True) -> None:
        self.cfg = cfg
        self._api = _init_ptrade_api(cfg.data_path, cfg.market, quiet=quiet)
        if cfg.as_of:
            self._api.context.current_dt = pd.Timestamp(cfg.as_of).to_pydatetime()

    def _as_of_for_api(self) -> str | None:
        if self.cfg.as_of:
            return str(pd.Timestamp(self.cfg.as_of).date())
        return None

    def list_symbols(self) -> list[str]:
        return sorted(self._api.get_Ashares(self._as_of_for_api()))

    def get_stock_name(self, symbol: str) -> str:
        name = self._api.get_stock_name(symbol)
        return str(name).strip() if name else ""

    def load_ohlcv(self, symbol: str) -> pd.DataFrame:
        df = self._api.get_price(
            symbol,
            end_date=self._as_of_for_api(),
            count=self.cfg.params.window_trading_days,
            frequency="1d",
            fields=_OHLCV_FIELDS,
            fq=self.cfg.resolved_fq(),
        )
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=_OHLCV_FIELDS)
        return df
