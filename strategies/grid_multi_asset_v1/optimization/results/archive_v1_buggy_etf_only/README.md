# 归档说明：v1.0 原始优化结果（有 bug）

## 问题描述

本目录保存的是 2026-05-08 第一次优化运行的结果。

**该次优化存在严重 bug**：策略代码中 `get_fundamentals` 使用了错误的字段名：
- `pe_ratio` → 应为 `pe_ttm`
- `market_cap` → 应为 `total_value`

导致 `dropna` 清空了所有股票行，`_refresh_pool` 每次只选出 8 只候选 ETF，
优化器实际调的是一个"纯 8只ETF 网格策略"，而非设计中的"股票+ETF 50只混合策略"。

## 影响

- 1007 个 Trial 全部基于错误策略
- 最优参数（Trial 33）对真实策略无参考意义
- 需要在修复列名 bug 后重新优化

## 修复记录

Git commit: fix: correct SimTradeLab valuation column names pe_ratio->pe_ttm, market_cap->total_value
