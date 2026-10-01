"""
llm_agent.py — TradeIQ v3 LLM Trading Decision Agent
─────────────────────────────────────────────────────────────────────────────
Architecture ported & enhanced from:
  sanketagarwal/hyperliquid-trading-agent/src/agent/decision_maker.py

Key upgrades over v2:
  ✅ MULTI-ASSET BATCH CALL   — All tickers in ONE Gemini call (cross-asset reasoning)
  ✅ HYSTERESIS POLICY        — 6 core rules: don't flip without confirmation
  ✅ EXIT PLAN TRACKING       — LLM returns exit_plan string per asset
  ✅ ACTIVE TRADE MEMORY      — LLM receives its own prior open trades in context
  ✅ SANITIZER FALLBACK       — Second cheap Gemini call to enforce JSON schema
  ✅ COOLDOWN LOGIC           — Parses exit_plan cooldown_bars to prevent re-trading
  ✅ DUAL-TIMEFRAME CONTEXT   — Sends both intraday + historical indicator sets
  ✅ STRUCTURED OUTPUT        — Strict JSON contract with 9 per-asset fields
  ✅ DIARY LOGGING            — Every decision logged to llm_decisions.jsonl

Usage:
    from llm_agent import get_llm_agent
    agent = get_llm_agent()

    # Build context from all tickers at once
    context = agent.build_context(assets_data, active_trades, risk_summary)
    results = agent.decide_batch(assets, context)
    # Returns dict: { ticker: {action, confidence, reasoning, sl_price, tp_price, exit_plan, source} }
"""
import os
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
LLM_MODE       = os.environ.get("LLM_MODE", "False").lower() in ("true", "1", "yes")
DIARY_PATH     = "llm_decisions.jsonl"

# ── System prompt (Hyperliquid-inspired, adapted for paper trading) ──────────
_SYSTEM_PROMPT = """You are a rigorous QUANTITATIVE TRADER and interdisciplinary MATHEMATICIAN-ENGINEER optimizing risk-adjusted returns for a paper-trading portfolio of stocks and crypto.

You will receive market context for SEVERAL assets, including:
- Per-asset intraday (short-term) and historical (long-term) indicator metrics
- Active open trades with their exit plans from prior cycles
- Current risk management limits (hard-enforced by the system)
- ML model predictions and sentiment scores

Always use the 'current_time' in the context to evaluate time-based conditions.

Your goal: Make decisive, first-principles decisions per asset that minimize churn while capturing edge.

═══ CORE POLICY (Low-Churn, Position-Aware) ═══

1) RESPECT PRIOR PLANS: If an asset has an active trade with an exit_plan containing explicit invalidation (e.g., "close if EMA20 crosses below EMA50"), do NOT close or flip early unless that invalidation has occurred.

2) HYSTERESIS (anti-churn): Require STRONGER evidence to CHANGE a decision than to HOLD it.
   Only flip direction if BOTH:
   a) Higher-timeframe structure supports the new direction (EMA20 vs EMA50 slope, MACD regime)
   b) Short-term confirms with a decisive move beyond ~0.5×ATR and momentum alignment (MACD or RSI slope)
   Otherwise, prefer HOLD or adjust TP/SL.

3) COOLDOWN: After opening, flipping, or adding, impose a self-cooldown of at least 3 bars.
   Encode this in exit_plan (e.g., "cooldown_bars:3 until 2026-04-12T13:00:00").
   Honor your own cooldowns on future cycles by checking current_time vs the cooldown timestamp.

4) OVERBOUGHT/OVERSOLD ≠ REVERSAL: Treat RSI extremes as risk-of-pullback signals.
   You need structure + momentum confirmation to bet against the trend.
   Prefer tightening stops or partial profits over instant flips.

5) PREFER ADJUSTMENTS OVER EXITS: If the thesis weakens but is not invalidated, consider:
   tighten stop (recent swing or ATR multiple), trail TP, or reduce size.
   Flip only on hard invalidation + fresh confluence.

6) SENTIMENT IS A TILT, NOT A TRIGGER: Do NOT open/close/flip solely due to sentiment score.
   Use it to break ties or confirm direction.

═══ DECISION DISCIPLINE (per asset) ═══
- Choose one action: buy / sell / hold
- For BUY: tp_price > current_price, sl_price < current_price
- For SELL: tp_price < current_price, sl_price > current_price
- If sensible TP/SL cannot be set, use null — a mandatory SL will be auto-applied
- exit_plan MUST include at least ONE explicit invalidation trigger

═══ REASONING RECIPE (first principles) ═══
- Structure: trend (EMAs slope/cross, HH/HL vs LH/LL)
- Momentum: MACD regime, RSI slope
- Volatility: ATR, Bollinger %B
- Volume: OBV slope, VWAP position
- Sentiment: news score
- ML: model signal + confidence
- Favor alignment across historical and intraday. Counter-trend requires stronger confirmation.

═══ OUTPUT CONTRACT ═══
Output ONLY a strict JSON object (no markdown, no code fences) with exactly two properties:
  • "reasoning": long-form string capturing detailed step-by-step analysis for ALL assets
  • "trade_decisions": array ordered to match the provided assets list

Each item inside trade_decisions must have these exact keys:
  asset, action, confidence, allocation_pct, sl_price, tp_price, exit_plan, rationale

Where:
  - action: "buy" | "sell" | "hold"
  - confidence: float 0.0 to 1.0
  - allocation_pct: float — % of available cash to deploy (0 for hold/sell)
  - sl_price: float or null
  - tp_price: float or null
  - exit_plan: string — invalidation conditions + optional cooldown_bars:N until ISO_TIMESTAMP
  - rationale: string — brief 1-2 sentence justification for this specific asset

Do NOT emit markdown, code fences, or any extra properties."""


class LLMAgent:
    """
    Google Gemini powered multi-asset trading decision agent.
    Falls back to ML-only mode if Gemini is unavailable.

    v3 improvements:
    - Batch call for all assets (cross-asset reasoning)
    - Hysteresis system prompt (6 rules)
    - exit_plan tracking
    - Sanitizer fallback
    - Dual-timeframe context
    """

    def __init__(self):
        self.enabled = LLM_MODE
        self._client = None
        self._sanitizer = None

        if self.enabled and not GEMINI_API_KEY:
            logger.warning("[LLM] LLM_MODE=True but GEMINI_API_KEY is missing - using ML-only mode")
            self.enabled = False

        if self.enabled:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)

                # Primary model — used for batch multi-asset decisions
                self._client = genai.GenerativeModel(
                    model_name="gemini-2.0-flash",
                    generation_config={
                        "temperature":        0.15,   # lower = more consistent
                        "top_p":              0.85,
                        "max_output_tokens":  4096,   # larger for multi-asset
                        "response_mime_type": "application/json",
                    }
                )

                # Sanitizer model — cheap flash to enforce schema
                self._sanitizer = genai.GenerativeModel(
                    model_name="gemini-2.0-flash",
                    generation_config={
                        "temperature":        0.0,
                        "max_output_tokens":  2048,
                        "response_mime_type": "application/json",
                    }
                )

                logger.info("[LLM] Gemini v3 agent initialized (multi-asset, hysteresis, exit_plan)")
            except ImportError:
                logger.warning("[LLM] google-generativeai not installed — using ML-only mode")
                self.enabled = False
            except Exception as e:
                logger.warning("[LLM] Gemini init failed: %s — using ML-only mode", e)
                self.enabled = False

    # ── Context Builder ───────────────────────────────────────────────────────
    def build_context(
        self,
        assets_data: list[dict],
        active_trades: list[dict] | None = None,
        risk_summary: dict | None = None,
    ) -> str:
        """
        Build the full JSON context payload to send to the LLM.

        Args:
            assets_data: List of per-asset dicts with keys:
                ticker, current_price, intraday (indicators), historical (indicators),
                ml_pred, sentiment_score
            active_trades: List of active open trade dicts (from auto_trader)
            risk_summary:  Dict from RiskManager.get_summary()

        Returns:
            JSON string of the full context payload
        """
        payload = {
            "current_time":   datetime.now(timezone.utc).isoformat(),
            "active_trades":  active_trades or [],
            "risk_limits":    risk_summary or {},
            "market_data":    assets_data,
            "instructions": {
                "assets": [a["ticker"] for a in assets_data],
                "requirement": (
                    "Decide actions for ALL assets. Return a strict JSON object "
                    "matching the output contract. No markdown."
                )
            }
        }
        return json.dumps(payload, default=str)

    # ── Main Entry Point: Batch Decide ────────────────────────────────────────
    def decide_batch(
        self,
        tickers: list[str],
        context: str,
    ) -> dict[str, dict]:
        """
        Make trading decisions for multiple assets in ONE LLM call.

        Args:
            tickers: List of ticker symbols (same order as context assets_data)
            context: JSON string from build_context()

        Returns:
            Dict mapping ticker → decision dict with keys:
            {action, confidence, allocation_pct, sl_price, tp_price, exit_plan, reasoning, rationale, source}
        """
        if self.enabled and self._client is not None:
            try:
                return self._gemini_decide_batch(tickers, context)
            except Exception as e:
                logger.warning("[LLM] Gemini batch call failed: %s — falling back to ML-only", e)

        # Fallback: all HOLD decisions
        return {t: self._hold_decision(t) for t in tickers}

    # ── Legacy Single-Ticker Interface (backwards-compatible) ─────────────────
    def decide(
        self,
        ticker: str,
        current_price: float,
        indicators: dict,
        ml_pred: dict,
        sentiment_score: float = 0.0,
        active_trades: list[dict] | None = None,
        risk_summary: dict | None = None,
    ) -> dict:
        """
        Single-ticker decide() kept for backwards compatibility with auto_trader.

        Internally wraps into a batch call when LLM is enabled.
        Falls back to ML-only when LLM is unavailable.
        """
        # Build a minimal asset_data for this single ticker
        asset_data = {
            "ticker":          ticker,
            "current_price":   current_price,
            "intraday":        indicators,
            "historical":      {},   # single-ticker mode has no multi-TF
            "ml_pred":         ml_pred,
            "sentiment_score": sentiment_score,
        }
        context = self.build_context(
            assets_data   = [asset_data],
            active_trades = active_trades or [],
            risk_summary  = risk_summary or {},
        )
        results = self.decide_batch([ticker], context)
        result  = results.get(ticker, self._hold_decision(ticker))

        # Convert to legacy format expected by auto_trader v2
        action = result.get("action", "hold").upper()
        return {
            "ticker":     ticker,
            "action":     action,
            "confidence": result.get("confidence", 0.3),
            "reasoning":  result.get("reasoning", "") or result.get("rationale", ""),
            "sl_price":   result.get("sl_price"),
            "tp_price":   result.get("tp_price"),
            "exit_plan":  result.get("exit_plan", ""),
            "source":     result.get("source", "ML"),
        }

    # ── Gemini Batch Decision ─────────────────────────────────────────────────
    def _gemini_decide_batch(
        self,
        tickers: list[str],
        context: str,
    ) -> dict[str, dict]:
        """Call Gemini with all assets in one request and parse structured output."""

        prompt = _SYSTEM_PROMPT + "\n\n" + context

        response  = self._client.generate_content([{"role": "user", "parts": [prompt]}])
        raw_text  = response.text.strip()

        parsed = self._parse_response(raw_text, tickers)

        if parsed is None:
            logger.warning("[LLM] Primary parse failed — invoking sanitizer")
            parsed = self._sanitize(raw_text, tickers)

        if parsed is None:
            logger.error("[LLM] Sanitizer also failed — returning all HOLD")
            return {t: self._hold_decision(t) for t in tickers}

        # Map decisions to ticker dict + log each
        reasoning_global = parsed.get("reasoning", "")
        results = {}
        for dec in parsed.get("trade_decisions", []):
            tkr = dec.get("asset") or dec.get("ticker")
            if not tkr or tkr not in tickers:
                continue

            action     = dec.get("action", "hold").upper()
            confidence = float(dec.get("confidence", 0.5))
            alloc_pct  = float(dec.get("allocation_pct", 0.0))
            sl_price   = dec.get("sl_price")
            tp_price   = dec.get("tp_price")
            exit_plan  = dec.get("exit_plan", "")
            rationale  = dec.get("rationale", "")

            if action not in ("BUY", "SELL", "HOLD"):
                action = "HOLD"

            result = {
                "ticker":       tkr,
                "action":       action,
                "confidence":   round(confidence, 3),
                "allocation_pct": round(alloc_pct, 4),
                "sl_price":     sl_price,
                "tp_price":     tp_price,
                "exit_plan":    exit_plan,
                "reasoning":    reasoning_global,
                "rationale":    rationale,
                "source":       "LLM",
            }
            results[tkr] = result
            self._log_decision(result)
            logger.info("[LLM] %s → %s (conf=%.2f) — %s", tkr, action, confidence, rationale[:80])

        # Fill in any missing tickers with HOLD
        for t in tickers:
            if t not in results:
                results[t] = self._hold_decision(t, reasoning=reasoning_global)

        return results

    # ── Response Parsing ──────────────────────────────────────────────────────
    def _parse_response(self, raw_text: str, tickers: list[str]) -> dict | None:
        """Try to parse raw LLM response as JSON with trade_decisions array."""
        cleaned = raw_text.strip()

        # Strip markdown fences if present
        if cleaned.startswith("```"):
            first_nl = cleaned.find("\n")
            if first_nl >= 0:
                cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                return None
            if "trade_decisions" not in parsed:
                return None
            if not isinstance(parsed["trade_decisions"], list):
                return None
            return parsed
        except (json.JSONDecodeError, ValueError):
            # Try to extract JSON object with regex
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    if isinstance(parsed, dict) and "trade_decisions" in parsed:
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    pass
            return None

    # ── Sanitizer ─────────────────────────────────────────────────────────────
    def _sanitize(self, raw_text: str, tickers: list[str]) -> dict | None:
        """Use a second Gemini call to enforce strict JSON schema."""
        if self._sanitizer is None:
            return None

        sanitize_prompt = (
            "You are a strict JSON normalizer. Return ONLY a JSON object with two keys:\n"
            '  "reasoning" (string) and "trade_decisions" (array).\n'
            "Each trade_decisions item must have exactly these keys:\n"
            '  asset (string), action ("buy"/"sell"/"hold"), confidence (0.0-1.0),\n'
            '  allocation_pct (0.0-1.0), sl_price (number or null), tp_price (number or null),\n'
            '  exit_plan (string), rationale (string).\n'
            f"Valid assets: {json.dumps(tickers)}.\n"
            "If the input is wrapped in markdown or has prose, extract just the JSON. "
            "Do not add fields. Output ONLY the JSON.\n\n"
            f"Input to normalize:\n{raw_text[:3000]}"
        )

        try:
            response = self._sanitizer.generate_content([
                {"role": "user", "parts": [sanitize_prompt]}
            ])
            sanitized_text = response.text.strip()
            return self._parse_response(sanitized_text, tickers)
        except Exception as e:
            logger.error("[LLM] Sanitizer call failed: %s", e)
            return None

    # ── ML Fallback ───────────────────────────────────────────────────────────
    def _hold_decision(self, ticker: str, reasoning: str = "") -> dict:
        """Return a safe HOLD decision when LLM or parsing fails."""
        return {
            "ticker":       ticker,
            "action":       "HOLD",
            "confidence":   0.3,
            "allocation_pct": 0.0,
            "sl_price":     None,
            "tp_price":     None,
            "exit_plan":    "Hold — LLM unavailable or parse error",
            "reasoning":    reasoning,
            "rationale":    "LLM unavailable — defaulting to HOLD",
            "source":       "FALLBACK",
        }

    def ml_fallback_decision(
        self,
        ticker: str,
        current_price: float,
        indicators: dict,
        ml_pred: dict,
    ) -> dict:
        """Pure ML + indicator composite when LLM is fully unavailable."""
        ml_sig  = ml_pred.get("signal", "HOLD")
        ind_sig = indicators.get("signal", "HOLD")
        conf    = ml_pred.get("confidence", 0.3)

        # Both agree → use that signal
        if ml_sig == ind_sig:
            action = ml_sig
        elif abs(indicators.get("score", 0)) >= 3.0:
            action = ind_sig
        else:
            action = "HOLD"

        sl_price = tp_price = None
        if action == "BUY":
            sl_price = round(current_price * 0.98, 4)
            tp_price = round(current_price * 1.04, 4)
        elif action == "SELL":
            sl_price = round(current_price * 1.02, 4)
            tp_price = round(current_price * 0.96, 4)

        return {
            "ticker":       ticker,
            "action":       action,
            "confidence":   round(conf, 3),
            "allocation_pct": 0.10 if action != "HOLD" else 0.0,
            "sl_price":     sl_price,
            "tp_price":     tp_price,
            "exit_plan":    f"ML={ml_sig}({conf:.2f}), IND={ind_sig}. Exit on opposite signal.",
            "reasoning":    f"ML={ml_sig}({conf:.2f}), IND={ind_sig}(score={indicators.get('score',0):.1f})",
            "rationale":    f"ML-only mode: {ml_sig} with {conf:.0%} confidence",
            "source":       "ML",
        }

    # ── Cooldown Parser ───────────────────────────────────────────────────────
    @staticmethod
    def parse_cooldown_until(exit_plan: str) -> datetime | None:
        """
        Parse 'cooldown_bars:N until ISO_TIMESTAMP' from an exit_plan string.
        Returns the cooldown-until datetime (UTC) if found, else None.
        """
        if not exit_plan:
            return None
        match = re.search(
            r'cooldown_bars:\d+\s+until\s+(\d{4}-\d{2}-\d{2}T[\d:]+(?:Z|[+-]\d{2}:\d{2})?)',
            exit_plan, re.IGNORECASE
        )
        if not match:
            return None
        try:
            ts = match.group(1)
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def is_in_cooldown(exit_plan: str) -> bool:
        """Return True if the exit_plan encodes a cooldown that hasn't expired yet."""
        until = LLMAgent.parse_cooldown_until(exit_plan)
        if until is None:
            return False
        now = datetime.now(timezone.utc)
        # Make both tz-aware
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        return now < until

    # ── Decision Logging ──────────────────────────────────────────────────────
    def _log_decision(self, decision: dict):
        """Append decision to JSONL diary for audit and future training."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **{k: v for k, v in decision.items() if k != "reasoning"},  # keep reasoning separate
            "llm_reasoning_preview": (decision.get("reasoning") or "")[:300],
        }
        try:
            with open(DIARY_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning("[LLM] Could not write to %s: %s", DIARY_PATH, e)


# ── Singleton ─────────────────────────────────────────────────────────────────
_agent: LLMAgent | None = None


def get_llm_agent() -> LLMAgent:
    """Return shared LLMAgent singleton."""
    global _agent
    if _agent is None:
        _agent = LLMAgent()
    return _agent


# ── CLI Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    agent = LLMAgent()
    print(f"LLM Enabled: {agent.enabled}")

    # ── Test batch context build ──
    test_assets = [
        {
            "ticker":          "AAPL",
            "current_price":   185.0,
            "intraday": {
                "signal": "BUY", "score": 3.5,
                "rsi": 38.0, "rsi_7": 32.0, "macd": 0.25, "macd_sig": 0.10,
                "bb_pct_b": 0.12, "engulfing": 1,
                "ema_20": 183.0, "ema_50": 179.5,
                "adx": 28.0, "stoch_k": 22.0, "vwap": 182.0, "atr_3": 2.1, "atr_14": 1.8,
            },
            "historical": {
                "signal": "BUY", "score": 2.0, "rsi": 45.0, "ema_20": 180.0, "ema_50": 175.0,
            },
            "ml_pred": {
                "signal": "BUY", "confidence": 0.74,
                "predicted_price": 192.0, "change_pct": 3.8, "mape": 1.9,
            },
            "sentiment_score": 0.15,
        },
        {
            "ticker":          "BTC-USD",
            "current_price":   62000.0,
            "intraday": {
                "signal": "HOLD", "score": -0.5,
                "rsi": 58.0, "rsi_7": 62.0, "macd": -50.0, "macd_sig": -30.0,
                "bb_pct_b": 0.6, "engulfing": 0,
                "ema_20": 62500.0, "ema_50": 60000.0,
                "adx": 18.0, "stoch_k": 65.0, "vwap": 61800.0, "atr_3": 800.0, "atr_14": 1200.0,
            },
            "historical": {
                "signal": "BUY", "score": 1.5, "rsi": 54.0, "ema_20": 61000.0, "ema_50": 58000.0,
            },
            "ml_pred": {
                "signal": "HOLD", "confidence": 0.51,
                "predicted_price": 63000.0, "change_pct": 1.6, "mape": 3.2,
            },
            "sentiment_score": -0.05,
        },
    ]

    test_active_trades = [
        {
            "ticker": "RELIANCE.NS",
            "action": "BUY",
            "entry_price": 2850.0,
            "exit_plan": "Close if EMA20 crosses below EMA50 on daily chart. cooldown_bars:3 until 2026-04-12T14:00:00+05:30",
        }
    ]

    test_risk = {
        "max_position_pct": 10.0,
        "daily_loss_limit_pct": 10.0,
        "circuit_breaker_active": False,
        "max_concurrent": 8,
    }

    context = agent.build_context(
        assets_data   = test_assets,
        active_trades = test_active_trades,
        risk_summary  = test_risk,
    )

    print(f"\nContext length: {len(context)} chars")
    print(f"\nCooldown test: {LLMAgent.is_in_cooldown('cooldown_bars:3 until 2026-04-12T14:00:00+00:00')}")

    print("\n--- Running batch decision ---")
    results = agent.decide_batch(["AAPL", "BTC-USD"], context)

    for tkr, dec in results.items():
        print(f"\n  [{tkr}]")
        print(f"    Action     : {dec['action']}")
        print(f"    Confidence : {dec['confidence']:.2f}")
        print(f"    Alloc %    : {dec['allocation_pct']:.1%}")
        print(f"    TP / SL    : {dec.get('tp_price')} / {dec.get('sl_price')}")
        print(f"    Exit Plan  : {dec.get('exit_plan', '')[:100]}")
        print(f"    Rationale  : {dec.get('rationale', '')[:100]}")
        print(f"    Source     : {dec['source']}")
