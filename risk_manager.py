"""
risk_manager.py — TradeIQ Hardened Risk Manager
Ported from: sanketagarwal/hyperliquid-trading-agent/src/risk_manager.py
Adapted for: Paper-trading mode with SQLite portfolio

All safety guards are enforced here, independent of ML/LLM decisions.
Neither the ML model nor the LLM can override these limits — they are
hard-coded checks applied before every trade execution.

Configurable via .env:
    MAX_POSITION_PCT         = 10   (% of portfolio per trade)
    STOP_LOSS_PCT            = 2    (auto stop-loss % from entry)
    TAKE_PROFIT_PCT          = 4    (auto take-profit % from entry)
    MAX_CONCURRENT           = 8    (max open positions at once)
    DAILY_LOSS_LIMIT         = 10   (circuit breaker % drawdown)
    MAX_TOTAL_EXPOSURE_PCT   = 50   (total notional exposure %)
    MANDATORY_SL_PCT         = 2    (force-set SL if missing)
    BALANCE_RESERVE_PCT      = 20   (minimum % of starting capital to keep)
    MAX_FORCE_CLOSE_LOSS_PCT = 20   (force-close at this loss %)
"""
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Load config from environment / defaults ───────────────────────────────────
def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except Exception:
        return default

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except Exception:
        return default


class RiskManager:
    """
    Enforces risk limits on every trade before execution.
    Hard-coded safety — cannot be overridden by ML or LLM.
    """

    def __init__(self):
        self.max_position_pct         = _env_float("MAX_POSITION_PCT", 10.0)
        self.stop_loss_pct            = _env_float("STOP_LOSS_PCT", 2.0)
        self.take_profit_pct          = _env_float("TAKE_PROFIT_PCT", 4.0)
        self.max_concurrent           = _env_int("MAX_CONCURRENT", 8)
        self.daily_loss_limit_pct     = _env_float("DAILY_LOSS_LIMIT", 10.0)
        self.max_total_exposure_pct   = _env_float("MAX_TOTAL_EXPOSURE_PCT", 50.0)
        self.mandatory_sl_pct         = _env_float("MANDATORY_SL_PCT", 2.0)
        self.balance_reserve_pct      = _env_float("BALANCE_RESERVE_PCT", 20.0)
        self.max_force_close_loss_pct = _env_float("MAX_FORCE_CLOSE_LOSS_PCT", 20.0)
        self.max_leverage             = _env_float("MAX_LEVERAGE", 1.0)

        # Daily tracking
        self._daily_high_value   = None
        self._daily_high_date    = None
        self._circuit_breaker_on = False

    # ── Daily drawdown tracking ───────────────────────────────────────────────
    def _update_daily_high(self, capital: float):
        today = datetime.now(timezone.utc).date()
        if self._daily_high_date != today:
            self._daily_high_value   = capital
            self._daily_high_date    = today
            self._circuit_breaker_on = False
            logger.info("[RISK] New trading day — circuit breaker reset")
        elif capital > self._daily_high_value:
            self._daily_high_value = capital

    # ── Individual checks ─────────────────────────────────────────────────────

    def check_circuit_breaker(self, current_capital: float) -> tuple[bool, str]:
        """Block all new trades if daily drawdown exceeds limit."""
        self._update_daily_high(current_capital)
        if self._circuit_breaker_on:
            return False, "Daily loss circuit breaker is ACTIVE — no new trades allowed today"
        if self._daily_high_value and self._daily_high_value > 0:
            dd_pct = (self._daily_high_value - current_capital) / self._daily_high_value * 100
            if dd_pct >= self.daily_loss_limit_pct:
                self._circuit_breaker_on = True
                msg = (f"Circuit breaker TRIGGERED: drawdown {dd_pct:.2f}% "
                       f"exceeds limit {self.daily_loss_limit_pct}%")
                logger.warning(f"[RISK] {msg}")
                return False, msg
        return True, ""

    def check_balance_reserve(self, current_capital: float, starting_capital: float) -> tuple[bool, str]:
        """Don't trade if capital falls below minimum reserve."""
        if starting_capital <= 0:
            return True, ""
        min_capital = starting_capital * (self.balance_reserve_pct / 100.0)
        if current_capital < min_capital:
            msg = (f"Capital {current_capital:.2f} below reserve "
                   f"{min_capital:.2f} ({self.balance_reserve_pct}% of starting capital)")
            return False, msg
        return True, ""

    def check_position_size(self, trade_value: float, current_capital: float) -> tuple[bool, float, str]:
        """
        Caps individual trade at max_position_pct of portfolio.
        Returns (allowed, capped_value, reason).
        If trade_value > cap, it is automatically reduced (not rejected).
        """
        if current_capital <= 0:
            return False, 0, "Capital is zero"
        max_trade = current_capital * (self.max_position_pct / 100.0)
        if trade_value > max_trade:
            logger.warning(
                "[RISK] Position size capped: %.2f → %.2f (%.0f%% limit)",
                trade_value, max_trade, self.max_position_pct
            )
            return True, max_trade, f"Trade capped to {self.max_position_pct}% of portfolio"
        return True, trade_value, ""

    def check_total_exposure(self, new_trade: float, open_positions_value: float,
                              current_capital: float) -> tuple[bool, str]:
        """Total exposure (all open positions + new trade) cannot exceed max_total_exposure_pct."""
        total = open_positions_value + new_trade
        max_exp = current_capital * (self.max_total_exposure_pct / 100.0)
        if total > max_exp:
            msg = (f"Total exposure {total:.2f} would exceed "
                   f"{self.max_total_exposure_pct}% of capital ({max_exp:.2f})")
            return False, msg
        return True, ""

    def check_concurrent_positions(self, current_open: int) -> tuple[bool, str]:
        """Limit number of simultaneous open positions."""
        if current_open >= self.max_concurrent:
            msg = f"Already at max concurrent positions ({self.max_concurrent})"
            return False, msg
        return True, ""

    def check_leverage(self, open_positions_value: float, new_trade_value: float, current_capital: float) -> tuple[bool, str]:
        """Block trades that would push the portfolio leverage over max allowed."""
        if current_capital <= 0:
            return False, "Capital is zero"
        total_implied_exposure = open_positions_value + new_trade_value
        implied_leverage = total_implied_exposure / current_capital
        if implied_leverage > self.max_leverage:
             msg = f"Trade would exceed max leverage {self.max_leverage}x (implied: {implied_leverage:.2f}x)"
             return False, msg
        return True, ""

    # ── Auto Stop-Loss / Take-Profit ──────────────────────────────────────────
    def compute_sl_tp(self, entry_price: float, is_buy: bool) -> dict:
        """
        Auto-compute stop-loss and take-profit levels if not provided by ML/LLM.
        Returns {'sl_price': float, 'tp_price': float}
        """
        sl_dist = entry_price * (self.stop_loss_pct / 100.0)
        tp_dist = entry_price * (self.take_profit_pct / 100.0)
        if is_buy:
            return {
                "sl_price": round(entry_price - sl_dist, 4),
                "tp_price": round(entry_price + tp_dist, 4),
            }
        else:
            return {
                "sl_price": round(entry_price + sl_dist, 4),
                "tp_price": round(entry_price - tp_dist, 4),
            }

    # ── Force-close check ─────────────────────────────────────────────────────
    def positions_to_force_close(self, positions: list[dict]) -> list[dict]:
        """
        Given a list of open position dicts with keys:
            ticker, quantity, entry_price
        and current prices provided inline as 'current_price',
        returns list of positions that should be force-closed due to excessive loss.
        """
        to_close = []
        for pos in positions:
            ticker      = pos.get("ticker", "?")
            entry       = float(pos.get("entry_price", 0))
            qty         = float(pos.get("quantity", 0))
            current_px  = float(pos.get("current_price", entry))

            if entry <= 0 or qty <= 0:
                continue

            pnl_pct = (current_px - entry) / entry * 100
            if pnl_pct <= -self.max_force_close_loss_pct:
                loss = (current_px - entry) * qty
                logger.warning(
                    "[RISK] Force-close %s: loss %.2f%% (%.2f) exceeds max %.2f%%",
                    ticker, abs(pnl_pct), loss, self.max_force_close_loss_pct
                )
                to_close.append({
                    "ticker":     ticker,
                    "quantity":   qty,
                    "entry_price": entry,
                    "current_price": current_px,
                    "loss_pct":   round(pnl_pct, 2),
                    "loss_amount": round(loss, 2),
                })
        return to_close

    # ── Composite validation ──────────────────────────────────────────────────
    def validate_buy(self,
                     ticker: str,
                     trade_value: float,
                     current_capital: float,
                     starting_capital: float,
                     open_positions_count: int,
                     open_positions_value: float) -> tuple[bool, float, str]:
        """
        Run all pre-trade checks for a BUY order.
        Returns (allowed, capped_trade_value, reason).
        reason is "" if allowed.
        """
        # 1. Circuit breaker
        ok, reason = self.check_circuit_breaker(current_capital)
        if not ok:
            return False, 0, reason

        # 2. Balance reserve
        ok, reason = self.check_balance_reserve(current_capital, starting_capital)
        if not ok:
            return False, 0, reason

        # 3. Concurrent positions
        ok, reason = self.check_concurrent_positions(open_positions_count)
        if not ok:
            return False, 0, reason

        # 4. Position size (auto-caps, never rejects)
        ok, capped_value, reason = self.check_position_size(trade_value, current_capital)
        if not ok:
            return False, 0, reason
        if reason:
            logger.info("[RISK] %s: %s", ticker, reason)

        # 5. Total exposure
        ok, reason = self.check_total_exposure(capped_value, open_positions_value, current_capital)
        if not ok:
            return False, 0, reason

        # 6. Leverage check
        ok, reason = self.check_leverage(open_positions_value, capped_value, current_capital)
        if not ok:
            return False, 0, reason

        return True, capped_value, ""

    def get_summary(self) -> dict:
        """Return current risk config as a dict (for logging / LLM context)."""
        return {
            "max_position_pct":         self.max_position_pct,
            "stop_loss_pct":            self.stop_loss_pct,
            "take_profit_pct":          self.take_profit_pct,
            "max_concurrent":           self.max_concurrent,
            "daily_loss_limit_pct":     self.daily_loss_limit_pct,
            "max_total_exposure_pct":   self.max_total_exposure_pct,
            "circuit_breaker_active":   self._circuit_breaker_on,
            "max_force_close_loss_pct": self.max_force_close_loss_pct,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_risk_manager = None

def get_risk_manager() -> RiskManager:
    """Return shared RiskManager singleton."""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager


# ── CLI Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rm = RiskManager()
    print("Risk Manager Config:")
    for k, v in rm.get_summary().items():
        print(f"  {k}: {v}")

    # Test BUY validation
    ok, capped, reason = rm.validate_buy(
        ticker="AAPL",
        trade_value=15000,
        current_capital=100000,
        starting_capital=100000,
        open_positions_count=3,
        open_positions_value=20000,
    )
    print(f"\nTest BUY $15000 from $100K capital:")
    print(f"  Allowed: {ok}  Capped to: {capped:.2f}  Reason: '{reason}'")

    # Test SL/TP computation
    sl_tp = rm.compute_sl_tp(entry_price=150.0, is_buy=True)
    print(f"\nAuto SL/TP for BUY @ 150: {sl_tp}")
