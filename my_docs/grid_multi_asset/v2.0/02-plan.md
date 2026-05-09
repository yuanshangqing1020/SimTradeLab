# 多标的自适应网格策略 v2.0 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 v1 基础上，新增大盘趋势判断（BULL/NEUTRAL/BEAR）和层数感知仓位管理（单标的权重上限 + 趋势投入比例），完成 Walk-Forward 调参后移植到 JoinQuant 平台。

**Architecture:** 最小侵入式改造 —— 新增纯函数 `_calc_regime` / `_apply_weight_cap`，修改 `_detect_regime`（API包装层）和 `_execute_grid`（资金分配），其余逻辑完全不变。SimTradeLab 负责本地调参，JoinQuant 负责平台回测验证。

**Tech Stack:** Python 3.8+, NumPy, Pandas, Optuna, SimTradeLab, JoinQuant（聚宽）

---

## 文件清单

| 操作 | 路径 | 说明 |
|---|---|---|
| 修复 | `tests/unit/test_grid_multi_asset.py:17` | 路径 `grid_multi_asset` → `grid_multi_asset_v1` |
| 创建 | `strategies/grid_multi_asset_v2/template.py` | v2 策略模板（调参注入用） |
| 创建 | `strategies/grid_multi_asset_v2/backtest.py` | v2 直接回测（含最优参数，调参后填写） |
| 创建 | `strategies/grid_multi_asset_v2/optimization/optimize_params.py` | Walk-Forward 调参入口 |
| 创建 | `strategies/grid_multi_asset_v2/optimization/results/.gitkeep` | 占位，保证目录存在 |
| 创建 | `strategies/grid_multi_asset_v2/stats/.gitkeep` | 占位，保证目录存在 |
| 创建 | `tests/unit/test_grid_multi_asset_v2.py` | v2 新增 8 个单元测试 |
| 修改 | `strategies_jq/grid_multi_asset/README.md` | 版本对照表新增 v2 行 |
| 创建 | `strategies_jq/grid_multi_asset/v2/strategy.py` | JQ 平台版（Task 8，调参完成后） |

---

## Task 1：修复 v1 测试路径 + 确认 v1 测试通过

**Files:**
- Modify: `tests/unit/test_grid_multi_asset.py:17`

- [ ] **Step 1: 修改路径**

将第 17 行改为：

```python
_STRATEGY_PATH = Path(__file__).parents[2] / 'strategies' / 'grid_multi_asset_v1' / 'backtest.py'
```

- [ ] **Step 2: 运行 v1 测试确认全部通过**

```bash
cd /mnt/c/Quant-Workspace/SimTradeLab
conda run -n SimTrade python -m pytest tests/unit/test_grid_multi_asset.py -v
```

期望输出：`24 passed`（所有测试绿灯）

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_grid_multi_asset.py
git commit -m "fix: correct v1 test strategy path to grid_multi_asset_v1"
```

---

## Task 2：创建 v2 目录结构

**Files:**
- Create: `strategies/grid_multi_asset_v2/optimization/results/.gitkeep`
- Create: `strategies/grid_multi_asset_v2/stats/.gitkeep`

- [ ] **Step 1: 创建目录和占位文件**

```bash
cd /mnt/c/Quant-Workspace/SimTradeLab
mkdir -p strategies/grid_multi_asset_v2/optimization/results
mkdir -p strategies/grid_multi_asset_v2/stats
touch strategies/grid_multi_asset_v2/optimization/results/.gitkeep
touch strategies/grid_multi_asset_v2/stats/.gitkeep
```

- [ ] **Step 2: 确认目录结构**

```bash
find strategies/grid_multi_asset_v2 -type f
```

期望输出：
```
strategies/grid_multi_asset_v2/optimization/results/.gitkeep
strategies/grid_multi_asset_v2/stats/.gitkeep
```

---

## Task 3：TDD — 写失败测试

**Files:**
- Create: `tests/unit/test_grid_multi_asset_v2.py`

- [ ] **Step 1: 创建测试文件**

```python
# tests/unit/test_grid_multi_asset_v2.py
# -*- coding: utf-8 -*-
"""
v2 新增功能单元测试：
  - _calc_regime：大盘趋势判断（纯数学，不依赖 API）
  - _apply_weight_cap：单标的权重截断（纯数学）
"""
import types
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# ── 加载策略纯函数 ──────────────────────────────────────────────────────────── #
_STRATEGY_PATH = Path(__file__).parents[2] / 'strategies' / 'grid_multi_asset_v2' / 'template.py'

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

_fns         = _load_fns()
_calc_regime = _fns['_calc_regime']
_apply_cap   = _fns['_apply_weight_cap']
_normalize   = _fns['_normalize_weights']


# ── _calc_regime ────────────────────────────────────────────────────────────── #
class TestCalcRegime:
    def _make_prices(self, n=260, trend='flat'):
        """生成测试用价格序列。trend: 'up'/'down'/'flat'"""
        np.random.seed(0)
        base = np.ones(n) * 3000.0
        if trend == 'up':
            base += np.linspace(0, 500, n)   # 明确上升趋势
        elif trend == 'down':
            base -= np.linspace(0, 500, n)   # 明确下降趋势
        return base

    def test_bull_when_above_both_ma(self):
        prices = self._make_prices(260, 'up')
        assert _calc_regime(prices) == 'BULL'

    def test_bear_when_below_both_ma(self):
        prices = self._make_prices(260, 'down')
        assert _calc_regime(prices) == 'BEAR'

    def test_neutral_when_between_ma(self):
        # 先下跌后反弹：价格在 MA120 上方但仍低于 MA250
        prices = self._make_prices(260, 'down')
        prices[-60:] += 300  # 近期反弹，越过 MA120 但未越过 MA250
        result = _calc_regime(prices)
        assert result == 'NEUTRAL'

    def test_neutral_on_short_history(self):
        prices = np.ones(100) * 3000.0  # 不足 250 条
        assert _calc_regime(prices) == 'NEUTRAL'

    def test_neutral_on_empty_array(self):
        assert _calc_regime(np.array([])) == 'NEUTRAL'


# ── _apply_weight_cap ───────────────────────────────────────────────────────── #
class TestApplyWeightCap:
    def test_no_overflow_after_cap(self):
        # 4只标的，等权 0.25，cap 设为 0.20
        raw = [0.25, 0.25, 0.25, 0.25]
        result = _apply_cap(raw, max_w=0.20)
        assert all(w <= 0.20 + 1e-9 for w in result), f"weight overflow: {result}"

    def test_sum_equals_one_after_cap(self):
        raw = [0.40, 0.30, 0.20, 0.10]
        result = _apply_cap(raw, max_w=0.25)
        assert abs(sum(result) - 1.0) < 1e-9

    def test_idempotent_after_extra_iteration(self):
        raw = [0.40, 0.30, 0.20, 0.10]
        result3 = _apply_cap(raw, max_w=0.25, iterations=3)
        result6 = _apply_cap(result3, max_w=0.25, iterations=3)
        for a, b in zip(result3, result6):
            assert abs(a - b) < 1e-9, "权重未收敛"

    def test_no_cap_needed_unchanged(self):
        # 权重已低于 cap，不应改变
        raw = [0.10, 0.20, 0.30, 0.40]
        result = _apply_cap(raw, max_w=0.50)
        for a, b in zip(raw, result):
            assert abs(a - b) < 1e-9
```

- [ ] **Step 2: 运行测试，确认全部失败（template.py 尚不存在）**

```bash
cd /mnt/c/Quant-Workspace/SimTradeLab
conda run -n SimTrade python -m pytest tests/unit/test_grid_multi_asset_v2.py -v 2>&1 | tail -10
```

期望输出：包含 `ERROR` 或 `FileNotFoundError`（template.py 不存在）

---

## Task 4：实现 template.py（v2 核心）

**Files:**
- Create: `strategies/grid_multi_asset_v2/template.py`

- [ ] **Step 1: 创建 template.py**

```python
# strategies/grid_multi_asset_v2/template.py
# -*- coding: utf-8 -*-
"""
多标的自适应网格策略 v2

新增功能（相比 v1）：
  1. _calc_regime: 纯数学大盘趋势判断（BULL/NEUTRAL/BEAR）
  2. _detect_regime: API包装，更新 context.regime / context.invested_ratio
  3. _apply_weight_cap: 单标的权重截断再归一化
  4. _execute_grid: 使用 invested_ratio + 权重上限替代固定满仓逻辑

参数：由 optimization/optimize_params.py Walk-Forward 自动调参
"""
import numpy as np

# ── ETF 候选池（固定，与 v1 相同）────────────────────────────────────────────── #
CANDIDATE_ETFS = [
    '510300.SS', '510500.SS', '159915.SZ', '512880.SS', '512690.SS',
    '512010.SS', '515050.SS', '512480.SS', '159949.SZ', '588000.SS',
    '512170.SS', '512760.SS', '159792.SZ', '513100.SS', '513050.SS',
]

TARGET_CAPITAL = 500000.0  # 策略目标资金规模（绝对上限）


def initialize(context):
    set_benchmark('000300.SS')
    set_slippage(slippage=0.00246)

    # ── 沿用 v1 参数（optimizer 通过 context.* regex 注入）────────────────── #
    context.MAX_HOLD             = 10    # 最多持仓标的数
    context.GRID_STEP_VOL_FACTOR = 0.45  # 步长 = clip(vol * factor, min, max)
    context.GRID_STEP_MIN        = 0.01  # 步长下限
    context.GRID_STEP_MAX        = 0.05  # 步长上限
    context.GRID_MAX_LAYER       = 2     # 最大偏离层数
    context.LAYER_FRACTION       = 0.08  # 每层权重增减幅度
    context.VOL_WEIGHT           = 0.50  # 波动率在综合打分中的权重
    context.REBALANCE_FREQ       = 10    # 重新选股间隔（交易日）

    # ── v2 新增参数 ──────────────────────────────────────────────────────── #
    context.BULL_RATIO    = 0.80  # 牛市总投入比例
    context.NEUTRAL_RATIO = 0.60  # 震荡总投入比例
    context.BEAR_RATIO    = 0.35  # 熊市总投入比例

    # ── 运行时状态 ──────────────────────────────────────────────────────── #
    context.pool           = []         # 当前活跃网格池
    context.day_counter    = 0          # 交易日计数器
    context.regime         = 'NEUTRAL'  # 大盘状态
    context.invested_ratio = context.NEUTRAL_RATIO  # 当前投入比例


def handle_data(context, data):
    context.day_counter += 1
    if context.day_counter == 1 or context.day_counter % context.REBALANCE_FREQ == 0:
        _detect_regime(context)   # 先判断大盘趋势
        _refresh_pool(context)    # 再选股换仓
    _execute_grid(context)


def after_trading_end(context, data):
    held = sum(1 for p in context.portfolio.positions.values() if p.amount > 0)
    log.info('日终 | %s | 投入%.0f%% | 总资产: %.0f | 网格池: %d只 | 持仓: %d只 | 现金: %.0f' % (
        context.regime,
        context.invested_ratio * 100,
        context.portfolio.portfolio_value,
        len(context.pool),
        held,
        context.portfolio.cash,
    ))


# ── 纯数学函数（无 PTrade 依赖，可单元测试）─────────────────────────────────── #

def _calc_vol_from_prices(prices):
    """计算年化已实现波动率。（与 v1 完全相同）"""
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
    """计算当前网格层数。（与 v1 完全相同）"""
    raw = (ma20 - price) / (price * step)
    return int(np.clip(int(np.floor(raw + 0.5)), -max_layer, max_layer))


def _normalize_weights(raw_weights):
    """将原始权重列表归一化到 sum=1。（与 v1 完全相同）"""
    if not raw_weights:
        return []
    total = sum(raw_weights)
    if total <= 0:
        n = len(raw_weights)
        return [1.0 / n] * n
    return [w / total for w in raw_weights]


def _calc_regime(prices):
    """【v2 新增】纯数学：根据沪深300价格序列判断大盘趋势状态。

    Input:  numpy array，建议至少 250 根 K 线
    Output: 'BULL' / 'NEUTRAL' / 'BEAR'

    规则：
      价格 > MA120 且 > MA250 → BULL
      价格 < MA120 且 < MA250 → BEAR
      其他（含数据不足）       → NEUTRAL
    """
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 250:
        return 'NEUTRAL'
    price_now = arr[-1]
    ma120 = arr[-120:].mean()
    ma250 = arr[-250:].mean()
    above_120 = price_now > ma120
    above_250 = price_now > ma250
    if above_120 and above_250:
        return 'BULL'
    if (not above_120) and (not above_250):
        return 'BEAR'
    return 'NEUTRAL'


def _apply_weight_cap(norm_w, max_w, iterations=3):
    """【v2 新增】截断单标的最大权重并再归一化，迭代至收敛。

    Input:  norm_w    - list of floats（归一化权重，sum ≈ 1.0）
            max_w     - float，单标的权重上限
            iterations - 迭代次数（3次足够收敛）
    Output: list of floats（sum = 1.0，每项 ≤ max_w）
    """
    result = list(norm_w)
    for _ in range(iterations):
        clipped = [min(w, max_w) for w in result]
        result = _normalize_weights(clipped)
    return result


def _score_universe(vol_dict, fund_df, etf_codes, vol_weight):
    """综合打分。（与 v1 完全相同）"""
    import pandas as pd

    if not vol_dict:
        return []

    etf_set = set(etf_codes)
    records = []

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
            pe   = float(row['pe_ttm'])    if 'pe_ttm'    in fd.columns else None
            roe  = float(row['roe'])        if 'roe'        in fd.columns else 0.0
            mcap = float(row['total_value']) if 'total_value' in fd.columns else None
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
    df['qual_pct'] = 0.5

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

def _detect_regime(context):
    """【v2 新增】拉取沪深300历史，调用 _calc_regime，更新 context。
    仅在换股日调用，避免每日重复拉 260 根 K 线。
    """
    try:
        hist = get_history(260, '1d', 'close', ['000300.SS'])
        prices = hist['000300.SS'].dropna().values
    except Exception as exc:
        log.warning('_detect_regime get_history 失败: %s，保持当前状态' % str(exc))
        return

    context.regime = _calc_regime(prices)
    ratio_map = {
        'BULL':    context.BULL_RATIO,
        'NEUTRAL': context.NEUTRAL_RATIO,
        'BEAR':    context.BEAR_RATIO,
    }
    context.invested_ratio = ratio_map[context.regime]
    log.info('大盘状态: %s | 投入比例: %.0f%%' % (
        context.regime, context.invested_ratio * 100))


def _refresh_pool(context):
    """重新选股。（与 v1 完全相同）"""
    stocks = list(set(
        get_index_stocks('000300.SS') + get_index_stocks('000905.SS')
    ))
    etfs = list(CANDIDATE_ETFS)
    all_cands = stocks + etfs

    if not all_cands:
        log.warning('候选池为空，保留原池')
        return

    st_map   = get_stock_status(all_cands, 'ST')
    halt_map = get_stock_status(all_cands, 'HALT')
    stocks = [s for s in stocks
              if not st_map.get(s, False) and not halt_map.get(s, False)]
    etfs = [e for e in etfs
            if not st_map.get(e, False) and not halt_map.get(e, False)]

    if not stocks and not etfs:
        log.warning('ST/停牌过滤后候选池为空，保留原池')
        return

    fund_df = None
    if stocks:
        try:
            raw = get_fundamentals(stocks, 'valuation', ['pe_ttm', 'total_value', 'roe'])
            if raw is not None and len(raw) > 0:
                if 'code' not in raw.columns and raw.index.name == 'code':
                    raw = raw.reset_index()
                raw = raw.dropna(subset=['pe_ttm', 'total_value'])
                raw = raw[(raw['pe_ttm'] > 0) & (raw['pe_ttm'] < 120)]
                raw = raw[raw['total_value'] >= 3e9]
                if 'code' in raw.columns:
                    stocks = [s for s in stocks if s in raw['code'].values]
                fund_df = raw
        except Exception as exc:
            log.warning('get_fundamentals 失败: %s，跳过基本面过滤' % str(exc))

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

    ranked = _score_universe(vol_dict, fund_df, etfs, context.VOL_WEIGHT)
    new_pool = [code for code, _ in ranked[:context.MAX_HOLD]]

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
    """【v2 修改】每日收盘前执行网格，使用 invested_ratio + 单标的权重上限。"""
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
        step = float(np.clip(
            vol * context.GRID_STEP_VOL_FACTOR,
            context.GRID_STEP_MIN,
            context.GRID_STEP_MAX,
        ))
        layer = _calc_layer(price, ma20, step, context.GRID_MAX_LAYER)
        layers.append(layer)
        active.append(code)

    if not active:
        return

    # 原始权重（网格层数加权）
    raw_w  = [max((1.0 / N) * (1.0 + context.LAYER_FRACTION * float(lyr)), 1e-9)
              for lyr in layers]
    norm_w = _normalize_weights(raw_w)

    # 【v2 新增】单标的权重上限 = 等权基础 × (1 + 最大超配)
    max_w = (1.0 / N) * (1.0 + context.LAYER_FRACTION * context.GRID_MAX_LAYER)
    norm_w = _apply_weight_cap(norm_w, max_w)

    # 【v2 新增】趋势感知总投入上限
    tv  = context.portfolio.portfolio_value
    cap = tv * context.invested_ratio
    cap = min(cap, TARGET_CAPITAL)  # 绝对上限，防超规模

    for code, w in zip(active, norm_w):
        order_target_value(code, cap * w)
```

- [ ] **Step 2: 运行 v2 测试，确认全部通过**

```bash
conda run -n SimTrade python -m pytest tests/unit/test_grid_multi_asset_v2.py -v
```

期望输出：`9 passed`（5个 regime + 4个 weight_cap）

- [ ] **Step 3: 同时运行 v1 测试，确认未受影响**

```bash
conda run -n SimTrade python -m pytest tests/unit/test_grid_multi_asset.py tests/unit/test_grid_multi_asset_v2.py -v
```

期望输出：`33 passed`（24 + 9）

- [ ] **Step 4: Commit**

```bash
git add strategies/grid_multi_asset_v2/template.py \
        strategies/grid_multi_asset_v2/optimization/results/.gitkeep \
        strategies/grid_multi_asset_v2/stats/.gitkeep \
        tests/unit/test_grid_multi_asset_v2.py
git commit -m "feat: add grid_multi_asset_v2 template with regime detection and weight cap"
```

---

## Task 5：创建 backtest.py

**Files:**
- Create: `strategies/grid_multi_asset_v2/backtest.py`

backtest.py 与 template.py 内容完全相同，唯一区别是注释头标注"含最优参数"。调参完成前先用合理默认值，调参完成后按 Task 8 更新。

- [ ] **Step 1: 复制 template.py 为 backtest.py，修改头部注释**

```bash
cp strategies/grid_multi_asset_v2/template.py strategies/grid_multi_asset_v2/backtest.py
```

然后将 `backtest.py` 第 1-3 行改为：

```python
# strategies/grid_multi_asset_v2/backtest.py
# -*- coding: utf-8 -*-
"""
多标的自适应网格策略 v2 — 直接回测版

参数来源：Walk-Forward 调参最优值（调参完成后填写，当前为默认值）
初始资金：50 万
回测区间：2019-01-01 ~ 今日
"""
```

- [ ] **Step 2: 运行一次简单回测确认策略可正常启动**

修改 `src/simtradelab/backtest/run_backtest.py` 中配置，将 `strategy_name` 改为 `'grid_multi_asset_v2'`，起止日期改为 `'2025-01-01'` ~ `'2025-06-30'`（短期验证）：

```bash
conda run -n SimTrade python src/simtradelab/backtest/run_backtest.py
```

期望：策略正常启动，日志输出 `大盘状态: BULL/NEUTRAL/BEAR | 投入比例: XX%`，无 Python 错误。

- [ ] **Step 3: 恢复 run_backtest.py 配置**（改回 v1 或保持 v2，不影响调参）

- [ ] **Step 4: Commit**

```bash
git add strategies/grid_multi_asset_v2/backtest.py
git commit -m "feat: add grid_multi_asset_v2 backtest.py with default params"
```

---

## Task 6：创建 optimization/optimize_params.py

**Files:**
- Create: `strategies/grid_multi_asset_v2/optimization/optimize_params.py`

- [ ] **Step 1: 创建调参脚本**

```python
# strategies/grid_multi_asset_v2/optimization/optimize_params.py
# -*- coding: utf-8 -*-
"""
多标的自适应网格策略 v2 — Walk-Forward 参数优化器

参数空间: 5×3×2×2×3×3×3×3×3×3×3 = 11,664 组合
  （v1 基础上新增 BULL_RATIO/NEUTRAL_RATIO/BEAR_RATIO，
   MAX_HOLD 候选值加密为 5/8/10/12/15）
优化期: 2019-01-01 ~ 2024-12-31
留存期: 2025-01-01 ~ 2026-03-31（与 v1 相同口径，便于直接对比）

运行方式:
    cd /mnt/c/Quant-Workspace/SimTradeLab
    conda run -n SimTrade python strategies/grid_multi_asset_v2/optimization/optimize_params.py

断点续传: 直接重新运行，Optuna 从 results/optuna_journal.log 恢复
"""

from simtradelab.backtest.optimizer_framework import (
    ParameterSpace,
    optimize_strategy,
)


class GridMultiAssetV2Params(ParameterSpace):
    """v2 可调参数空间。

    参数空间大小: 5×3×2×2×3×3×3×3×3×3×3 = 11,664 组合
    Early-stopping patience: ~2916 次无改进后自动停止
    """

    MAX_HOLD             = [5, 8, 10, 12, 15]
    GRID_STEP_VOL_FACTOR = [0.30, 0.45, 0.60]
    GRID_STEP_MIN        = [0.01, 0.02]
    GRID_STEP_MAX        = [0.03, 0.05]
    GRID_MAX_LAYER       = [2, 3, 4]
    LAYER_FRACTION       = [0.08, 0.12, 0.16]
    VOL_WEIGHT           = [0.50, 0.65, 0.80]
    REBALANCE_FREQ       = [5, 10, 20]
    BULL_RATIO           = [0.70, 0.80, 0.90]
    NEUTRAL_RATIO        = [0.50, 0.60, 0.70]
    BEAR_RATIO           = [0.25, 0.35, 0.45]

    @staticmethod
    def validate(params):
        """拒绝无效参数组合。"""
        if params['GRID_STEP_MIN'] >= params['GRID_STEP_MAX']:
            raise ValueError(
                'GRID_STEP_MIN={} 必须小于 GRID_STEP_MAX={}'.format(
                    params['GRID_STEP_MIN'], params['GRID_STEP_MAX'],
                )
            )
        if not (params['BEAR_RATIO'] < params['NEUTRAL_RATIO'] < params['BULL_RATIO']):
            raise ValueError(
                'BEAR_RATIO < NEUTRAL_RATIO < BULL_RATIO 约束违反: '
                '{} / {} / {}'.format(
                    params['BEAR_RATIO'], params['NEUTRAL_RATIO'], params['BULL_RATIO'],
                )
            )
        return params


if __name__ == '__main__':
    custom_mapping = {
        'MAX_HOLD':             'context.MAX_HOLD',
        'GRID_STEP_VOL_FACTOR': 'context.GRID_STEP_VOL_FACTOR',
        'GRID_STEP_MIN':        'context.GRID_STEP_MIN',
        'GRID_STEP_MAX':        'context.GRID_STEP_MAX',
        'GRID_MAX_LAYER':       'context.GRID_MAX_LAYER',
        'LAYER_FRACTION':       'context.LAYER_FRACTION',
        'VOL_WEIGHT':           'context.VOL_WEIGHT',
        'REBALANCE_FREQ':       'context.REBALANCE_FREQ',
        'BULL_RATIO':           'context.BULL_RATIO',
        'NEUTRAL_RATIO':        'context.NEUTRAL_RATIO',
        'BEAR_RATIO':           'context.BEAR_RATIO',
    }

    optimize_strategy(
        parameter_space=GridMultiAssetV2Params,
        optimization_period=('2019-01-01', '2024-12-31'),
        holdout_period=('2025-01-01', '2026-03-31'),
        initial_capital=500000.0,
        walk_forward_config={
            'train_months': 24,
            'test_months':  6,
            'step_months':  6,
        },
        regularization_weight=0.1,
        stability_weight=0.5,
        custom_mapping=custom_mapping,
        resume=True,
        verbose=False,
        strategy_file='template.py',
    )
```

- [ ] **Step 2: 确认脚本可以被 import（语法检查）**

```bash
conda run -n SimTrade python -c "
import sys; sys.path.insert(0, '.')
# 仅做 class 定义部分的语法检查，不实际运行优化
exec(open('strategies/grid_multi_asset_v2/optimization/optimize_params.py').read().split(\"if __name__\")[0])
print('语法检查通过')
"
```

期望输出：`语法检查通过`

- [ ] **Step 3: Commit**

```bash
git add strategies/grid_multi_asset_v2/optimization/optimize_params.py
git commit -m "feat: add grid_multi_asset_v2 Walk-Forward optimizer with 11-param space"
```

---

## Task 7：更新 JoinQuant README

**Files:**
- Modify: `strategies_jq/grid_multi_asset/README.md`

- [ ] **Step 1: 在版本对照表中新增 v2 行**

在 `README.md` 的版本对照表（`## 版本对照表`）中，在 v1 行之后追加：

```markdown
| v2 | `strategies/grid_multi_asset_v2/` | Walk-Forward 调参（待完成）| 新增大盘趋势过滤（BULL/NEUTRAL/BEAR）+ 单标的权重上限 + 三档仓位比例 | 2026-05-09 |
```

- [ ] **Step 2: Commit**

```bash
git add strategies_jq/grid_multi_asset/README.md
git commit -m "docs: add v2 entry to grid_multi_asset JQ version table"
```

---

## Task 8：启动 Walk-Forward 调参（长时间任务）

**预估耗时：30~48 小时**

- [ ] **Step 1: 启动调参（后台运行）**

```bash
cd /mnt/c/Quant-Workspace/SimTradeLab
nohup conda run -n SimTrade python strategies/grid_multi_asset_v2/optimization/optimize_params.py \
    > strategies/grid_multi_asset_v2/stats/optimize_log.txt 2>&1 &
echo "PID: $!"
```

- [ ] **Step 2: 监控进度（任意时间检查）**

```bash
python3 - << 'EOF'
import json
trials = {}
journal = 'strategies/grid_multi_asset_v2/optimization/results/optuna_journal.log'
try:
    with open(journal) as f:
        for line in f:
            d = json.loads(line)
            if d.get('op_code') == 6 and d.get('state') == 2:
                trials[d['trial_id']] = d['values'][0]
    if trials:
        best_tid = max(trials, key=trials.get)
        print(f"已完成: {len(trials)} trials，最佳 Trial {best_tid}: {trials[best_tid]:.4f}")
    else:
        print("尚无完成的 trial")
except FileNotFoundError:
    print("journal 文件尚不存在，调参可能刚启动")
EOF
```

- [ ] **Step 3: 调参完成后，查看最优参数**

```bash
# 调参脚本会自动生成：
cat strategies/grid_multi_asset_v2/optimization/results/optimized_strategy.py | head -40
```

---

## Task 9：填入最优参数 + 运行 Holdout 回测

（在 Task 8 调参完成后执行）

**Files:**
- Modify: `strategies/grid_multi_asset_v2/backtest.py`（initialize 中的参数值）

- [ ] **Step 1: 将最优参数填入 backtest.py 的 initialize 函数**

从 `optimization/results/optimized_strategy.py` 读取 `context.*` 参数值，更新 `backtest.py` 对应行。例如（实际值以调参结果为准）：

```python
context.MAX_HOLD             = <最优值>
context.GRID_STEP_VOL_FACTOR = <最优值>
context.GRID_STEP_MIN        = <最优值>
context.GRID_STEP_MAX        = <最优值>
context.GRID_MAX_LAYER       = <最优值>
context.LAYER_FRACTION       = <最优值>
context.VOL_WEIGHT           = <最优值>
context.REBALANCE_FREQ       = <最优值>
context.BULL_RATIO           = <最优值>
context.NEUTRAL_RATIO        = <最优值>
context.BEAR_RATIO           = <最优值>
```

- [ ] **Step 2: 运行完整 Holdout 回测（2025-01-01 ~ 2026-03-31）**

修改 `src/simtradelab/backtest/run_backtest.py`：

```python
strategy_name    = 'grid_multi_asset_v2'
start_date       = '2025-01-01'
end_date         = '2026-03-31'
initial_capital  = 500000.0
```

```bash
conda run -n SimTrade python src/simtradelab/backtest/run_backtest.py
```

记录输出中的：年化收益率、夏普比率、最大回撤，与 v1 对比（v1 基准：+60.51% / 2.20 / -16.28%）。

- [ ] **Step 3: 与 v1 对比，确认成功标准**

| 指标 | v1 基准 | v2 目标 |
|---|---|---|
| 年化收益 | +60.51% | ≥ v1，或回撤改善可接受小幅下降 |
| 夏普比率 | 2.20 | 维持或提升 |
| 最大回撤 | -16.28% | < -16.28%（改善） |
| 首日仓位（NEUTRAL）| ~100% | < 70% |

- [ ] **Step 4: Commit**

```bash
git add strategies/grid_multi_asset_v2/backtest.py
git commit -m "feat: fill v2 backtest.py with Walk-Forward optimal params"
```

---

## Task 10：编写调参总结报告

**Files:**
- Create: `my_docs/grid_multi_asset/v2.0/03-optimization-summary.md`

参照 `v1.0/03-optimization-summary.md` 格式，记录：
1. 调参运行统计（总 trial 数、完成数、剪枝数、耗时）
2. 最优参数（Trial X）及参数规律解读
3. Walk-Forward 综合得分（与 v1 对比）
4. Holdout 回测结果（与 v1 对比）
5. 与 v1 的关键差异分析

- [ ] **Step 1: 填写报告并 commit**

```bash
git add my_docs/grid_multi_asset/v2.0/03-optimization-summary.md
git commit -m "docs: add grid_multi_asset v2.0 optimization summary"
```

---

## Task 11：移植到 JoinQuant 平台

（在 Task 9 Holdout 结果满意后执行）

**Files:**
- Create: `strategies_jq/grid_multi_asset/v2/strategy.py`

- [ ] **Step 1: 创建 v2 JQ 目录**

```bash
mkdir -p strategies_jq/grid_multi_asset/v2
```

- [ ] **Step 2: 基于 backtest.py 按 API 映射表移植**

以下替换需要全文搜索替换（逐条确认）：

| SimTradeLab | JoinQuant |
|---|---|
| `context.` | `g.` |
| `'000300.SS'` | `'000300.XSHG'` |
| `'000905.SS'` | `'000905.XSHG'` |
| `'.SS'` 结尾的代码 | `.XSHG` 结尾 |
| `'.SZ'` 结尾的代码 | `.XSHE` 结尾 |
| `get_history(N, '1d', 'close', codes)` | `history(N, '1d', 'close', codes, df=True)` |
| `get_fundamentals(stocks, 'valuation', [...])` | `get_fundamentals(query(valuation).filter(...))` |
| `pe_ttm` | `pe_ratio` |
| `total_value >= 3e9` | `market_cap >= 30`（单位亿元） |
| `p.amount` | `p.total_amount` |
| `portfolio.portfolio_value` | `portfolio.total_value` |
| `handle_data` 日频 | `run_daily(func, time='14:50')` |
| `set_slippage(slippage=X)` | `set_slippage(PriceRelatedSlippage(X))` |

- [ ] **Step 3: 在 JoinQuant 平台验证**

1. 打开 [JoinQuant 回测编辑器](https://www.joinquant.com/algorithm/index/edit)
2. 粘贴 `v2/strategy.py` 全文
3. 设置：起止日期 2025-01-01 ~ 2026-03-31，初始资金 50 万
4. 运行回测，与 SimTradeLab Holdout 结果对比

- [ ] **Step 4: 更新 JQ README 版本对照表**，将 v2 行的"待完成"替换为实际调参结果

- [ ] **Step 5: Commit**

```bash
git add strategies_jq/grid_multi_asset/v2/strategy.py \
        strategies_jq/grid_multi_asset/README.md
git commit -m "feat: add grid_multi_asset v2 JoinQuant strategy"
```

---

## 快速参考

**运行全量测试：**
```bash
cd /mnt/c/Quant-Workspace/SimTradeLab
conda run -n SimTrade python -m pytest tests/unit/test_grid_multi_asset.py tests/unit/test_grid_multi_asset_v2.py -v
```

**查看调参进度：**
```bash
python3 -c "
import json; trials = {}
[trials.update({d['trial_id']: d['values'][0]}) for line in open('strategies/grid_multi_asset_v2/optimization/results/optuna_journal.log') for d in [json.loads(line)] if d.get('op_code')==6 and d.get('state')==2]
best = max(trials, key=trials.get) if trials else None
print(f'完成 {len(trials)} trials，最佳 Trial {best}: {trials.get(best, \"N/A\"):.4f}' if best else '暂无完成 trial')
"
```
