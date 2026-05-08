# 多标的自适应网格策略 - 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 SimTradeLab 的 `strategies/grid_multi_asset/` 目录下实现多标的自适应网格策略，并配套 Optuna Walk-Forward 自动调参脚本。

**Architecture:** `backtest.py` 实现纯数学函数（可单元测试）+ PTrade API 调用封装；`optimization/optimize_params.py` 通过 `optimizer_framework.optimize_strategy` 注入参数并执行 Walk-Forward 优化。

**Tech Stack:** Python 3.12+、numpy、pandas、SimTradeLab PTrade API、optuna（已在 simtradelab[optimizer]）

**Spec:** `my_docs/2026-05-07-grid-multi-asset-design.md`

---

## 文件结构

| 文件 | 操作 | 职责 |
|---|---|---|
| `strategies/grid_multi_asset/backtest.py` | 创建 | 主策略：纯数学函数 + API 封装 + handle_data |
| `strategies/grid_multi_asset/optimization/optimize_params.py` | 创建 | Optuna 调参脚本 |
| `tests/unit/test_grid_multi_asset.py` | 创建 | 纯数学函数单元测试 |

---

## Task 1：策略骨架

**Files:**
- Create: `strategies/grid_multi_asset/backtest.py`

- [ ] **Step 1.1：创建目录并写入完整骨架文件**

```python
# strategies/grid_multi_asset/backtest.py
# -*- coding: utf-8 -*-
"""
多标的自适应网格策略

- 资金规模: 50万（TARGET_CAPITAL 软约束上限）
- 持仓数量: 10~50只（由 context.MAX_HOLD 控制）
- 标的: 沪深300+中证500 动态成分股 + 固定 ETF 候选池
- 网格步长: clip(vol * FACTOR, MIN, MAX)，区间 1%~5%
- 参数: 由 optimization/optimize_params.py Walk-Forward 自动调参
"""
import numpy as np

# ── ETF 候选池（固定）─────────────────────────────────────────────────────── #
CANDIDATE_ETFS = [
    '510300.SS', '510500.SS', '159915.SZ', '512880.SS', '512690.SS',
    '512010.SS', '515050.SS', '512480.SS', '159949.SZ', '588000.SS',
    '512170.SS', '512760.SS', '159792.SZ', '513100.SS', '513050.SS',
]

TARGET_CAPITAL = 500_000.0  # 策略目标资金规模（网格分配上限）


def initialize(context):
    set_benchmark('000300.SS')
    set_slippage(slippage=0.00246)

    # ── 可调参数（optimizer 通过 context.* regex 注入）────────────────── #
    context.MAX_HOLD             = 20    # 最多持仓标的数
    context.GRID_STEP_VOL_FACTOR = 0.45  # 步长 = clip(vol * factor, min, max)
    context.GRID_STEP_MIN        = 0.01  # 步长下限 1%
    context.GRID_STEP_MAX        = 0.04  # 步长上限 4%
    context.GRID_MAX_LAYER       = 3     # 最大偏离层数
    context.LAYER_FRACTION       = 0.12  # 每层权重增减幅度 ±12%
    context.VOL_WEIGHT           = 0.62  # 波动率在综合打分中的权重
    context.REBALANCE_FREQ       = 5     # 重新选股间隔（交易日）

    # ── 运行时状态 ──────────────────────────────────────────────────────── #
    context.pool        = []  # 当前活跃网格池（股票代码列表）
    context.day_counter = 0   # 交易日计数器


def handle_data(context, data):
    context.day_counter += 1
    if context.day_counter == 1 or context.day_counter % context.REBALANCE_FREQ == 0:
        _refresh_pool(context)
    _execute_grid(context)


def after_trading_end(context, data):
    held = sum(1 for p in context.portfolio.positions.values() if p.amount > 0)
    log.info('日终 | 总资产: %.0f | 网格池: %d只 | 持仓: %d只 | 现金: %.0f' % (
        context.portfolio.portfolio_value,
        len(context.pool),
        held,
        context.portfolio.cash,
    ))


# ── 纯数学函数（无 PTrade 依赖，可单元测试）─────────────────────────────────── #

def _calc_vol_from_prices(prices):
    """计算年化已实现波动率。
    Input:  numpy 数组，至少 22 根 K 线（1根用于第一个 ret）
    Output: float（年化 vol）或 None（数据不足/vol 为零）
    """
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 22:
        return None
    rets = np.diff(arr) / arr[:-1]
    valid = rets[np.isfinite(rets)]
    if len(valid) < 20:
        return None
    vol = float(valid[-20:].std() * np.sqrt(250.0))
    return vol if vol > 0 else None


def _calc_layer(price, ma20, step, max_layer):
    """计算当前网格层数。
    layer > 0: 价格低于中枢（超配信号）
    layer < 0: 价格高于中枢（欠配信号）
    layer = 0: 价格在中枢附近
    """
    raw = (ma20 - price) / (price * step)
    return int(np.clip(int(np.floor(raw + 0.5)), -max_layer, max_layer))


def _normalize_weights(raw_weights):
    """将原始权重列表归一化到 sum=1。
    若总和 <= 0，则返回等权；输入为空则返回空列表。
    """
    if not raw_weights:
        return []
    total = sum(raw_weights)
    if total <= 0:
        n = len(raw_weights)
        return [1.0 / n] * n
    return [w / total for w in raw_weights]


def _score_universe(vol_dict, fund_df, etf_codes, vol_weight):
    """综合打分，返回按得分降序排列的 [(code, score), ...]。

    vol_dict:   {code: annualized_vol}
    fund_df:    DataFrame with columns ['code','pe_ratio','market_cap','roe']，可为 None
    etf_codes:  ETF 代码列表（只用 vol + 流动性，无基本面）
    vol_weight: 波动率在总分中的权重（0~1）
    """
    import pandas as pd

    if not vol_dict:
        return []

    etf_set = set(etf_codes)
    records = []

    # 股票侧：波动率 + 基本面
    stock_codes = [c for c in vol_dict if c not in etf_set]
    if stock_codes and fund_df is not None and len(fund_df) > 0:
        if 'code' in fund_df.columns:
            fd = fund_df.set_index('code')
        else:
            fd = fund_df
        for code in stock_codes:
            if code not in vol_dict or code not in fd.index:
                continue
            row = fd.loc[code]
            pe   = float(row['pe_ratio'])   if 'pe_ratio'   in fd.columns else None
            roe  = float(row['roe'])        if 'roe'        in fd.columns else 0.0
            mcap = float(row['market_cap']) if 'market_cap' in fd.columns else None
            if pe is None or mcap is None:
                continue
            if not (np.isfinite(pe) and 0 < pe < 120):
                continue
            if not (np.isfinite(mcap) and mcap >= 3e9):
                continue
            roe = roe if np.isfinite(roe) else 0.0
            records.append({
                'code': code, 'kind': 'stock',
                'vol': vol_dict[code],
                'roe': roe,
                'inv_pe': 1.0 / max(pe, 1.0),
                'mcap': mcap,
            })

    # ETF 侧：只用波动率
    for code in etf_codes:
        if code in vol_dict:
            records.append({
                'code': code, 'kind': 'etf',
                'vol': vol_dict[code],
                'roe': 0.0, 'inv_pe': 0.0, 'mcap': 0.0,
            })

    if not records:
        return []

    df = pd.DataFrame(records)
    df['vol_pct'] = df['vol'].rank(pct=True)
    df['qual_pct'] = 0.5  # ETF 默认值

    stock_mask = df['kind'] == 'stock'
    if stock_mask.any():
        stk = df[stock_mask]
        df.loc[stock_mask, 'qual_pct'] = (
            stk['roe'].rank(pct=True) * 0.45
            + stk['inv_pe'].rank(pct=True) * 0.35
            + stk['mcap'].rank(pct=True) * 0.20
        )

    df['score'] = df['vol_pct'] * vol_weight + df['qual_pct'] * (1.0 - vol_weight)
    df = df.sort_values('score', ascending=False)
    return list(zip(df['code'], df['score']))


# ── API 依赖函数（需在 SimTradeLab 运行环境中执行）────────────────────────────── #

def _refresh_pool(context):
    """重新选股：从动态指数成分+ETF中，按综合打分选 Top-MAX_HOLD。"""
    # 1. 获取候选列表
    stocks = list(set(
        get_index_stocks('000300.SS') + get_index_stocks('000905.SS')
    ))
    etfs = list(CANDIDATE_ETFS)
    all_cands = stocks + etfs

    if not all_cands:
        log.warning('候选池为空，保留原池')
        return

    # 2. 过滤 ST / 停牌
    st_map   = get_stock_status(all_cands, 'ST')
    halt_map = get_stock_status(all_cands, 'HALT')
    stocks = [s for s in stocks
              if not st_map.get(s, False) and not halt_map.get(s, False)]
    etfs = [e for e in etfs
            if not st_map.get(e, False) and not halt_map.get(e, False)]

    if not stocks and not etfs:
        log.warning('ST/停牌过滤后候选池为空，保留原池')
        return

    # 3. 基本面数据（股票侧）
    fund_df = None
    if stocks:
        try:
            raw = get_fundamentals(stocks, 'valuation', ['pe_ratio', 'market_cap', 'roe'])
            if raw is not None and len(raw) > 0:
                if 'code' not in raw.columns and raw.index.name == 'code':
                    raw = raw.reset_index()
                raw = raw.dropna(subset=['pe_ratio', 'market_cap'])
                raw = raw[(raw['pe_ratio'] > 0) & (raw['pe_ratio'] < 120)]
                raw = raw[raw['market_cap'] >= 3e9]
                if 'code' in raw.columns:
                    stocks = [s for s in stocks if s in raw['code'].values]
                fund_df = raw
        except Exception as exc:
            log.warning('get_fundamentals 失败: %s，跳过基本面过滤' % str(exc))

    # 4. 计算波动率（一次批量拉取，减少 API 调用次数）
    all_active = stocks + etfs
    if not all_active:
        log.warning('有效候选池为空，保留原池')
        return

    vol_dict = {}
    try:
        hist = get_history(26, '1d', 'close', all_active)
        if hist is not None and len(hist) > 0:
            for code in all_active:
                if code not in hist.columns:
                    continue
                prices = hist[code].dropna().values
                v = _calc_vol_from_prices(prices)
                if v is not None:
                    vol_dict[code] = v
    except Exception as exc:
        log.warning('get_history(vol) 失败: %s' % str(exc))

    if not vol_dict:
        log.warning('波动率计算全部失败，保留原池')
        return

    # 5. 综合打分，取 Top-N
    ranked = _score_universe(vol_dict, fund_df, etfs, context.VOL_WEIGHT)
    new_pool = [code for code, _ in ranked[:context.MAX_HOLD]]

    # 6. 清仓已调出标的
    old_set = set(context.pool)
    new_set = set(new_pool)
    for code in old_set - new_set:
        order_target(code, 0)
        log.info('调出网格池: %s' % code)

    context.pool = new_pool
    log.info('网格池更新 %d只: %s%s' % (
        len(context.pool),
        ','.join(context.pool[:5]),
        '...' if len(context.pool) > 5 else '',
    ))


def _execute_grid(context):
    """每日收盘前：计算各标的网格层数，归一化权重后 order_target_value。"""
    if not context.pool:
        return

    N = len(context.pool)

    try:
        hist = get_history(31, '1d', 'close', context.pool)
    except Exception as exc:
        log.warning('_execute_grid get_history 失败: %s' % str(exc))
        return

    layers = []
    active = []

    for code in context.pool:
        if hist is None or code not in hist.columns:
            continue
        prices = hist[code].dropna().values
        if len(prices) < 22:
            continue
        price = float(prices[-1])
        if not (np.isfinite(price) and price > 0):
            continue
        ma20  = float(prices[-20:].mean())
        vol   = _calc_vol_from_prices(prices)
        if vol is None:
            continue
        step  = float(np.clip(
            vol * context.GRID_STEP_VOL_FACTOR,
            context.GRID_STEP_MIN,
            context.GRID_STEP_MAX,
        ))
        layer = _calc_layer(price, ma20, step, context.GRID_MAX_LAYER)
        layers.append(layer)
        active.append(code)

    if not active:
        return

    raw_w  = [max((1.0 / N) * (1.0 + context.LAYER_FRACTION * float(lyr)), 1e-9)
              for lyr in layers]
    norm_w = _normalize_weights(raw_w)

    tv  = context.portfolio.portfolio_value
    cap = min(tv, max(TARGET_CAPITAL, 1000.0))
    for code, w in zip(active, norm_w):
        order_target_value(code, cap * w)
```

- [ ] **Step 1.2：验证文件语法（在 SimTradeLab 项目根目录执行）**

```bash
cd /mnt/c/Quant-Workspace/SimTradeLab
poetry run python -c "
import ast, pathlib
src = pathlib.Path('strategies/grid_multi_asset/backtest.py').read_text()
ast.parse(src)
print('syntax OK')
"
```

Expected: `syntax OK`

- [ ] **Step 1.3：Commit**

```bash
git add strategies/grid_multi_asset/backtest.py
git commit -m "feat(grid): add multi-asset adaptive grid strategy scaffold"
```

---

## Task 2：纯数学函数单元测试（先写失败的测试）

**Files:**
- Create: `tests/unit/test_grid_multi_asset.py`

- [ ] **Step 2.1：创建测试文件**

```python
# tests/unit/test_grid_multi_asset.py
# -*- coding: utf-8 -*-
"""
纯数学函数单元测试（_calc_vol_from_prices / _calc_layer /
_normalize_weights / _score_universe）

使用 exec() 在 mock PTrade 命名空间中加载策略文件，
避免 set_benchmark 等运行时 API 未定义的问题。
"""
import types
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# ── 加载策略纯函数 ──────────────────────────────────────────────────────────── #
_STRATEGY_PATH = Path(__file__).parents[2] / 'strategies' / 'grid_multi_asset' / 'backtest.py'

def _load_fns():
    """用 mock PTrade 全局量执行策略文件，返回其命名空间。"""
    _log = types.SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
    )
    mock_ns = {
        '__builtins__': __builtins__,
        'np': np,
        'set_benchmark':      lambda *a, **kw: None,
        'set_slippage':       lambda *a, **kw: None,
        'get_history':        lambda *a, **kw: pd.DataFrame(),
        'get_fundamentals':   lambda *a, **kw: pd.DataFrame(),
        'get_index_stocks':   lambda *a, **kw: [],
        'get_stock_status':   lambda *a, **kw: {},
        'order_target':       lambda *a, **kw: None,
        'order_target_value': lambda *a, **kw: None,
        'log': _log,
    }
    src = _STRATEGY_PATH.read_text(encoding='utf-8')
    exec(compile(src, str(_STRATEGY_PATH), 'exec'), mock_ns)
    return mock_ns

_fns = _load_fns()
_calc_vol   = _fns['_calc_vol_from_prices']
_calc_layer = _fns['_calc_layer']
_normalize  = _fns['_normalize_weights']
_score      = _fns['_score_universe']


# ── _calc_vol_from_prices ───────────────────────────────────────────────────── #
class TestCalcVolFromPrices:
    def test_returns_none_for_fewer_than_22_bars(self):
        prices = np.linspace(10, 11, 20)
        assert _calc_vol(prices) is None

    def test_returns_float_for_valid_data(self):
        np.random.seed(42)
        prices = np.cumprod(1 + np.random.normal(0.0005, 0.015, 30)) * 100
        result = _calc_vol(prices)
        assert isinstance(result, float) and result > 0

    def test_constant_prices_return_none(self):
        prices = np.full(30, 10.0)
        assert _calc_vol(prices) is None   # std=0 → vol=0 → returns None

    def test_annualized_range_is_plausible(self):
        # daily std ~1% → annualized ~sqrt(250)*0.01 ≈ 15.8%
        np.random.seed(0)
        daily_rets = np.random.normal(0, 0.01, 25)
        prices = np.cumprod(1 + daily_rets) * 100
        vol = _calc_vol(prices)
        assert vol is not None
        assert 0.05 < vol < 0.60

    def test_uses_last_20_returns(self):
        # 前面几根极高波动，后20根低波动 → vol 应该接近低波动估计
        np.random.seed(1)
        prices_noisy = np.cumprod(1 + np.random.normal(0, 0.10, 10)) * 100
        prices_calm  = np.cumprod(1 + np.random.normal(0, 0.005, 22)) * prices_noisy[-1]
        prices = np.concatenate([prices_noisy, prices_calm])
        vol = _calc_vol(prices)
        assert vol is not None and vol < 0.20   # calm segment dominates


# ── _calc_layer ─────────────────────────────────────────────────────────────── #
class TestCalcLayer:
    def test_price_below_ma_positive_layer(self):
        # MA=10, price=9, step=0.05 → raw=(10-9)/(9*0.05)=2.22 → floor(2.22+0.5)=2
        assert _calc_layer(price=9.0, ma20=10.0, step=0.05, max_layer=3) == 2

    def test_price_above_ma_negative_layer(self):
        # MA=10, price=11, step=0.05 → raw=(10-11)/(11*0.05)=-1.82 → floor(-1.82+0.5)=-2
        assert _calc_layer(price=11.0, ma20=10.0, step=0.05, max_layer=3) == -2

    def test_at_ma_zero_layer(self):
        assert _calc_layer(price=10.0, ma20=10.0, step=0.05, max_layer=3) == 0

    def test_clamped_to_positive_max(self):
        # Price very far below MA
        assert _calc_layer(price=1.0, ma20=10.0, step=0.01, max_layer=3) == 3

    def test_clamped_to_negative_max(self):
        assert _calc_layer(price=20.0, ma20=10.0, step=0.01, max_layer=3) == -3

    def test_asymmetry_small_step(self):
        # step=0.02, price=9.9 vs MA=10 → raw=0.1/(9.9*0.02)=0.505 → floor(1.0)=1
        assert _calc_layer(price=9.9, ma20=10.0, step=0.02, max_layer=5) == 1


# ── _normalize_weights ──────────────────────────────────────────────────────── #
class TestNormalizeWeights:
    def test_sums_to_one(self):
        result = _normalize([1.0, 2.0, 3.0])
        assert abs(sum(result) - 1.0) < 1e-9

    def test_proportions_preserved(self):
        result = _normalize([1.0, 3.0])
        assert abs(result[1] / result[0] - 3.0) < 1e-9

    def test_all_zeros_equal_weight(self):
        result = _normalize([0.0, 0.0, 0.0])
        assert result == pytest.approx([1/3, 1/3, 1/3])

    def test_single_element(self):
        assert _normalize([5.0]) == [1.0]

    def test_empty_returns_empty(self):
        assert _normalize([]) == []

    def test_negative_not_amplified(self):
        # 策略中 raw_w 已用 max(..., 1e-9) 保证非负，此处验证归一化本身
        result = _normalize([2.0, 0.0, 2.0])
        assert result == pytest.approx([0.5, 0.0, 0.5])


# ── _score_universe ─────────────────────────────────────────────────────────── #
class TestScoreUniverse:

    def _fund_df(self, codes):
        return pd.DataFrame({
            'code':       codes,
            'pe_ratio':   [20.0, 15.0, 30.0][:len(codes)],
            'market_cap': [1e11, 5e10, 2e10][:len(codes)],
            'roe':        [0.15, 0.20, 0.10][:len(codes)],
        })

    def test_returns_sorted_descending(self):
        vol = {'A': 0.3, 'B': 0.5, 'C': 0.2}
        result = _score(vol, self._fund_df(['A', 'B', 'C']), [], vol_weight=1.0)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_high_vol_wins_when_vol_weight_1(self):
        vol = {'ETF1': 0.4, 'ETF2': 0.2}
        result = _score(vol, None, ['ETF1', 'ETF2'], vol_weight=1.0)
        assert result[0][0] == 'ETF1'

    def test_empty_vol_dict_returns_empty(self):
        assert _score({}, None, [], vol_weight=0.7) == []

    def test_filters_negative_pe(self):
        vol = {'A': 0.3, 'B': 0.4}
        fd = pd.DataFrame({
            'code': ['A', 'B'],
            'pe_ratio': [-5.0, 20.0],
            'market_cap': [5e10, 5e10],
            'roe': [0.1, 0.15],
        })
        result = _score(vol, fd, [], vol_weight=0.6)
        codes = [c for c, _ in result]
        assert 'A' not in codes
        assert 'B' in codes

    def test_filters_small_cap(self):
        vol = {'A': 0.3, 'B': 0.4}
        fd = pd.DataFrame({
            'code': ['A', 'B'],
            'pe_ratio': [20.0, 20.0],
            'market_cap': [1e9, 5e10],   # A < 30亿门槛
            'roe': [0.1, 0.15],
        })
        result = _score(vol, fd, [], vol_weight=0.6)
        codes = [c for c, _ in result]
        assert 'A' not in codes

    def test_etf_included_without_fund_df(self):
        vol = {'ETF1': 0.35}
        result = _score(vol, None, ['ETF1'], vol_weight=0.7)
        assert len(result) == 1 and result[0][0] == 'ETF1'

    def test_mixed_stocks_and_etfs(self):
        vol = {'S1': 0.3, 'ETF1': 0.6}
        fd = pd.DataFrame({
            'code': ['S1'],
            'pe_ratio': [20.0],
            'market_cap': [5e10],
            'roe': [0.15],
        })
        result = _score(vol, fd, ['ETF1'], vol_weight=0.9)
        codes = [c for c, _ in result]
        assert 'S1' in codes and 'ETF1' in codes
```

- [ ] **Step 2.2：运行测试，确认全部失败（函数尚未实现）**

```bash
cd /mnt/c/Quant-Workspace/SimTradeLab
poetry run pytest tests/unit/test_grid_multi_asset.py -v 2>&1 | head -30
```

Expected: 多个 `FAILED` 或 `ERROR`（因为函数体是 pass 或文件未创建）

> 注意：如果 Task 1 已经写入完整实现而非 pass，测试可能直接通过。按顺序执行时先跑 Task 2 再执行 Task 3 补充实现。

- [ ] **Step 2.3：Commit 测试文件**

```bash
git add tests/unit/test_grid_multi_asset.py
git commit -m "test(grid): add unit tests for pure math functions"
```

---

## Task 3：验证纯数学函数实现通过测试

（`backtest.py` 中的完整函数实现已在 Task 1 Step 1.1 中提供，此步骤仅执行测试验证）

- [ ] **Step 3.1：运行全部单元测试**

```bash
cd /mnt/c/Quant-Workspace/SimTradeLab
poetry run pytest tests/unit/test_grid_multi_asset.py -v
```

Expected:
```
tests/unit/test_grid_multi_asset.py::TestCalcVolFromPrices::test_returns_none_for_fewer_than_22_bars PASSED
tests/unit/test_grid_multi_asset.py::TestCalcVolFromPrices::test_returns_float_for_valid_data PASSED
...（共 22 个 PASSED）
```

如有失败，根据报错调整 `backtest.py` 中对应函数，不改变测试文件。

- [ ] **Step 3.2：Commit 修复**

```bash
git add strategies/grid_multi_asset/backtest.py
git commit -m "fix(grid): make all pure-math unit tests pass"
```

---

## Task 4：集成验证——单次短周期回测

目的：确认策略在 SimTradeLab 完整运行环境中无崩溃，handle_data 正常触发。

- [ ] **Step 4.1：临时修改 run_backtest.py 指向新策略**

编辑 `src/simtradelab/backtest/run_backtest.py`，将：
```python
strategy_name = '5mv'
start_date = '2025-01-01'
end_date = '2025-10-31'
initial_capital=100000.0
```
改为：
```python
strategy_name = 'grid_multi_asset'
start_date = '2023-01-01'
end_date = '2023-03-31'       # 仅3个月，快速验证
initial_capital = 500000.0
```

- [ ] **Step 4.2：运行回测**

```bash
cd /mnt/c/Quant-Workspace/SimTradeLab
poetry run python src/simtradelab/backtest/run_backtest.py 2>&1 | tail -30
```

Expected（关键检查点）：
1. 无 `Exception` / `Traceback`
2. 日志中出现 `网格池更新` 字样（说明 `_refresh_pool` 执行成功）
3. 日志中出现 `日终 | 总资产` 字样
4. 最终 `total_return` 不是 0.0（说明有实际交易发生）

如果出现异常，根据报错 traceback 修复 `backtest.py`，然后重新运行。

- [ ] **Step 4.3：恢复 run_backtest.py**

```python
strategy_name = '5mv'
start_date = '2025-01-01'
end_date = '2025-10-31'
initial_capital = 100000.0
```

- [ ] **Step 4.4：Commit**

```bash
git add strategies/grid_multi_asset/backtest.py
git commit -m "feat(grid): complete multi-asset adaptive grid strategy"
```

---

## Task 5：创建调参脚本

**Files:**
- Create: `strategies/grid_multi_asset/optimization/optimize_params.py`

- [ ] **Step 5.1：创建优化器目录和脚本**

```python
# strategies/grid_multi_asset/optimization/optimize_params.py
# -*- coding: utf-8 -*-
"""
多标的自适应网格策略 - Walk-Forward 参数优化器

参数空间: 4×3×2×2×3×3×3×3 = 3,888 组合
优化期: 2019-01-01 ~ 2024-12-31（6年，覆盖多轮牛熊）
留存期: 2025-01-01 ~ 2026-03-31（样本外泛化验证）

运行方式:
    cd /mnt/c/Quant-Workspace/SimTradeLab
    poetry run python strategies/grid_multi_asset/optimization/optimize_params.py

断点续传: 直接重新运行，Optuna 自动从 results/optuna_journal.log 恢复
"""

from simtradelab.backtest.optimizer_framework import (
    ParameterSpace,
    optimize_strategy,
)


class GridMultiAssetParams(ParameterSpace):
    """可调参数空间。

    参数空间大小: 4×3×2×2×3×3×3×3 = 3,888 组合
    Early-stopping patience: ~972 次无改进后自动停止
    """

    MAX_HOLD             = [10, 20, 30, 50]
    GRID_STEP_VOL_FACTOR = [0.30, 0.45, 0.60]
    GRID_STEP_MIN        = [0.01, 0.02]
    GRID_STEP_MAX        = [0.03, 0.05]
    GRID_MAX_LAYER       = [2, 3, 4]
    LAYER_FRACTION       = [0.08, 0.12, 0.16]
    VOL_WEIGHT           = [0.50, 0.65, 0.80]
    REBALANCE_FREQ       = [5, 10, 20]

    @staticmethod
    def validate(params):
        """拒绝 GRID_STEP_MIN >= GRID_STEP_MAX 的无效组合。"""
        if params['GRID_STEP_MIN'] >= params['GRID_STEP_MAX']:
            raise ValueError(
                'GRID_STEP_MIN={} 必须小于 GRID_STEP_MAX={}'.format(
                    params['GRID_STEP_MIN'], params['GRID_STEP_MAX'],
                )
            )
        return params


if __name__ == '__main__':
    # optimizer_framework 默认用 g.{param_name} 替换，
    # 我们的策略用 context.* 存储参数，需指定 custom_mapping。
    custom_mapping = {
        'MAX_HOLD':             'context.MAX_HOLD',
        'GRID_STEP_VOL_FACTOR': 'context.GRID_STEP_VOL_FACTOR',
        'GRID_STEP_MIN':        'context.GRID_STEP_MIN',
        'GRID_STEP_MAX':        'context.GRID_STEP_MAX',
        'GRID_MAX_LAYER':       'context.GRID_MAX_LAYER',
        'LAYER_FRACTION':       'context.LAYER_FRACTION',
        'VOL_WEIGHT':           'context.VOL_WEIGHT',
        'REBALANCE_FREQ':       'context.REBALANCE_FREQ',
    }

    optimize_strategy(
        parameter_space=GridMultiAssetParams,
        optimization_period=('2019-01-01', '2024-12-31'),
        holdout_period=('2025-01-01', '2026-03-31'),
        initial_capital=500_000.0,
        walk_forward_config={
            'train_months': 24,   # 24个月训练窗口
            'test_months':  6,    # 6个月验证窗口
            'step_months':  6,    # 每次向前滑动6个月
        },
        regularization_weight=0.1,   # 边界参数惩罚权重
        stability_weight=0.5,         # 训练/测试不稳定性惩罚
        custom_mapping=custom_mapping,
        resume=True,     # 支持断点续传
        verbose=False,   # 关闭调试输出
    )
```

- [ ] **Step 5.2：验证语法**

```bash
cd /mnt/c/Quant-Workspace/SimTradeLab
poetry run python -c "
import ast, pathlib
src = pathlib.Path('strategies/grid_multi_asset/optimization/optimize_params.py').read_text()
ast.parse(src)
print('syntax OK')
"
```

Expected: `syntax OK`

- [ ] **Step 5.3：Commit**

```bash
git add strategies/grid_multi_asset/optimization/optimize_params.py
git commit -m "feat(grid): add Walk-Forward optimizer for grid_multi_asset"
```

---

## Task 6：验证 custom_mapping 正则替换生效

目的：确认 optimizer_framework 的参数注入能正确找到并替换 `context.MAX_HOLD = 20` 这一行。

- [ ] **Step 6.1：在 Python REPL 中验证替换逻辑**

```bash
cd /mnt/c/Quant-Workspace/SimTradeLab
poetry run python - <<'EOF'
from simtradelab.backtest.optimizer_framework import apply_parameter_replacement
from pathlib import Path

src = Path('strategies/grid_multi_asset/backtest.py').read_text()
custom_mapping = {
    'MAX_HOLD':             'context.MAX_HOLD',
    'GRID_STEP_VOL_FACTOR': 'context.GRID_STEP_VOL_FACTOR',
    'GRID_STEP_MIN':        'context.GRID_STEP_MIN',
    'GRID_STEP_MAX':        'context.GRID_STEP_MAX',
    'GRID_MAX_LAYER':       'context.GRID_MAX_LAYER',
    'LAYER_FRACTION':       'context.LAYER_FRACTION',
    'VOL_WEIGHT':           'context.VOL_WEIGHT',
    'REBALANCE_FREQ':       'context.REBALANCE_FREQ',
}
test_params = {
    'MAX_HOLD': 30,
    'GRID_STEP_VOL_FACTOR': 0.60,
    'GRID_STEP_MIN': 0.02,
    'GRID_STEP_MAX': 0.05,
    'GRID_MAX_LAYER': 4,
    'LAYER_FRACTION': 0.16,
    'VOL_WEIGHT': 0.80,
    'REBALANCE_FREQ': 10,
}
modified = apply_parameter_replacement(src, test_params, custom_mapping)

# 验证替换结果
for param, expected in [
    ('context.MAX_HOLD', '30'),
    ('context.GRID_STEP_VOL_FACTOR', '0.6'),
    ('context.GRID_STEP_MIN', '0.02'),
    ('context.GRID_STEP_MAX', '0.05'),
]:
    for line in modified.splitlines():
        if param in line and '=' in line:
            print(f'OK: {line.strip()}')
            break
    else:
        print(f'MISSING: {param}')
EOF
```

Expected: 8 行 `OK: context.XXX = YYY`

- [ ] **Step 6.2：若有 MISSING，检查 backtest.py 中对应行的缩进格式**

`apply_parameter_replacement` 的正则匹配 `^\s*context.XXX\s*=\s*...`，确保每行以空格+变量名开头（`initialize` 函数内的标准缩进）。

---

## Task 7：运行调参（正式）

此步骤为实际执行，耗时较长（视数据量，约数小时）。

- [ ] **Step 7.1：启动优化（支持随时 Ctrl+C 中断，再次运行自动续传）**

```bash
cd /mnt/c/Quant-Workspace/SimTradeLab
poetry run python strategies/grid_multi_asset/optimization/optimize_params.py
```

观察关键日志：
- `创建新的 Study: grid_multi_asset_optimization` — 首次运行
- `__PROGRESS__:N/3888` — 进度更新
- `发现更好的参数: X.XXXX（提升 Y.YYYY）` — 有改进时
- `连续 972 次无改进，提前停止` — 早停触发

- [ ] **Step 7.2：查看最优参数**

```bash
ls -lt strategies/grid_multi_asset/optimization/results/best_params_*.json | head -1 | awk '{print $NF}' | xargs cat
```

Expected: JSON 格式的最优参数，例如：
```json
{
  "MAX_HOLD": 30,
  "GRID_STEP_VOL_FACTOR": 0.45,
  "GRID_STEP_MIN": 0.01,
  "GRID_STEP_MAX": 0.05,
  ...
}
```

- [ ] **Step 7.3：查看留存期泛化结果**

调参脚本结束时自动打印：
```
====================================================================
留存期（2025）泛化测试
====================================================================
  total_return: X.XXXX
  annual_return: X.XXXX
  sharpe_ratio: X.XXXX
  max_drawdown: X.XXXX
```

- [ ] **Step 7.4：Commit 最优参数文件**

```bash
git add strategies/grid_multi_asset/optimization/results/best_params_*.json
git commit -m "chore(grid): save Walk-Forward optimized parameters"
```

---

## 自审清单

**Spec 覆盖：**
- [x] 起始资金 50万 → `initial_capital=500_000.0` + `TARGET_CAPITAL`
- [x] 交易标的 10~50只 → `MAX_HOLD` 候选值 [10,20,30,50]
- [x] 高波动优质股票+ETF → `_score_universe` 波动率+ROE/PE/市值打分
- [x] 每格 1%~5% → `GRID_STEP_MIN/MAX` 候选值 [0.01,0.02] / [0.03,0.05]
- [x] 自动调参 → Task 5~7 Optuna Walk-Forward
- [x] 所有参数通过 custom_mapping 注入 → Task 6 验证

**无 Placeholder：** 所有步骤包含完整代码或完整命令
**类型一致性：** `context.REBALANCE_FREQ` 在 backtest.py 和 custom_mapping 中完全对应
