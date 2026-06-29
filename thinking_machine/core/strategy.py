"""Strategy specification.

A ``StrategySpec`` is the machine-readable form of a trading idea.  The agent
translates plain English into one of these, and ``ai.agent.write_strategy`` then
serialises it into ``strategies/current_strategy.py`` as runnable Python - that
generated file is the "code" the Thinking Machine produces and overwrites.

Signal vocabulary (must match columns produced by ``indicators.compute_all``):
    fvg_bull, fvg_bear, ob_bull, ob_bear,
    sweep_high, sweep_low,
    ema_cross_up, ema_cross_down, price_above_ema, price_below_ema
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

LONG_SIGNALS = {"fvg_bull", "ob_bull", "sweep_low", "ema_cross_up", "price_above_ema"}
SHORT_SIGNALS = {"fvg_bear", "ob_bear", "sweep_high", "ema_cross_down", "price_below_ema"}
ALL_SIGNALS = LONG_SIGNALS | SHORT_SIGNALS


@dataclass
class StrategySpec:
    name: str = "untitled"
    # Entry triggers. A side fires when ALL of its signals are True on a bar
    # (confluence). An empty list disables that side.
    long_entry: list[str] = field(default_factory=list)
    short_entry: list[str] = field(default_factory=list)

    # Indicator parameters.
    ema_period: int = 20
    atr_period: int = 14
    sweep_lookback: int = 20

    # Risk model (distances expressed in ATR multiples; target as R multiple).
    stop_atr: float = 1.0
    rr: float = 2.0

    # Session / timing.
    rth_only: bool = True
    entry_after: str | None = None   # e.g. "09:30" -> only take entries at/after this time
    entry_before: str | None = None  # e.g. "11:00" -> only before this time
    max_trades_per_day: int = 1
    exit_on_session_close: bool = True

    def sides(self) -> list[str]:
        s = []
        if self.long_entry:
            s.append("long")
        if self.short_entry:
            s.append("short")
        return s

    def describe(self) -> str:
        parts = [f"'{self.name}'"]
        if self.long_entry:
            parts.append(f"LONG when [{' + '.join(self.long_entry)}]")
        if self.short_entry:
            parts.append(f"SHORT when [{' + '.join(self.short_entry)}]")
        parts.append(f"stop={self.stop_atr}xATR, target={self.rr}R")
        if self.entry_after or self.entry_before:
            parts.append(f"window={self.entry_after or 'open'}..{self.entry_before or 'close'}")
        parts.append(f"max {self.max_trades_per_day}/day")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)
