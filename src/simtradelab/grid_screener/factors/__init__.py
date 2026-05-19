"""可插拔筛选因子。"""

from simtradelab.grid_screener.factors.base import Factor
from simtradelab.grid_screener.factors.registry import FactorRegistry, default_registry

__all__ = ["Factor", "FactorRegistry", "default_registry"]
