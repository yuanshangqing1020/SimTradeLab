# 归档说明：第二次无效优化（缓存命中旧结果）

## 问题描述

第一次优化（archive_v1_buggy_etf_only）的 backtest_cache 中有 5745 个 .pkl 文件。
删除 optuna_journal.log 后重新启动优化，但 backtest_cache 未清空。

优化器的缓存 key 仅包含「参数组合 + 时间窗口」，不包含策略代码内容 hash，
导致新优化完全命中旧缓存，几分钟内跑完 1006 个 Trial，结果与第一次完全相同：
- Best Trial: 33，Score: -0.5790，参数完全一致

## 正确的重跑流程

1. 同时清空 optuna_journal.log 和 backtest_cache/*.pkl
2. 重新执行 optimize_params.py
