import os
from typing import Any

from dotenv import load_dotenv, set_key, find_dotenv

try:
    from kiteconnect import KiteConnect
except ImportError:  # pragma: no cover
    KiteConnect = None

load_dotenv()

KITE_API_KEY = os.getenv("ZERODHA_API_KEY", "").strip()
KITE_API_SECRET = os.getenv("ZERODHA_API_SECRET", "").strip()
KITE_ACCESS_TOKEN = os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()
KITE_USER_ID = os.getenv("ZERODHA_USER_ID", "").strip()


def _reload_settings() -> None:
    global KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN, KITE_USER_ID
    load_dotenv(override=True)
    KITE_API_KEY = os.getenv("ZERODHA_API_KEY", "").strip()
    KITE_API_SECRET = os.getenv("ZERODHA_API_SECRET", "").strip()
    KITE_ACCESS_TOKEN = os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()
    KITE_USER_ID = os.getenv("ZERODHA_USER_ID", "").strip()


def _env_path() -> str:
    path = find_dotenv(usecwd=True)
    if path:
        return path
    return os.path.join(os.path.dirname(__file__), ".env")


def _save_env_value(key: str, value: str) -> None:
    path = _env_path()
    if not os.path.exists(path):
        open(path, "a", encoding="utf-8").close()
    set_key(path, key, value or "")


def is_available() -> bool:
    return KiteConnect is not None


def is_configured() -> bool:
    return bool(is_available() and KITE_API_KEY and KITE_API_SECRET and KITE_ACCESS_TOKEN)


def get_kite_client() -> Any:
    if not is_available():
        raise ImportError("kiteconnect is not installed. Install it with `pip install kiteconnect`.")
    if not KITE_API_KEY:
        raise RuntimeError("Missing ZERODHA_API_KEY in environment.")
    client = KiteConnect(api_key=KITE_API_KEY)
    if KITE_ACCESS_TOKEN:
        client.set_access_token(KITE_ACCESS_TOKEN)
    return client


def build_login_url() -> str:
    client = get_kite_client()
    return client.login_url()


def save_zerodha_credentials(api_key: str | None = None,
                             api_secret: str | None = None,
                             access_token: str | None = None,
                             user_id: str | None = None) -> None:
    if api_key is not None:
        _save_env_value("ZERODHA_API_KEY", api_key.strip())
    if api_secret is not None:
        _save_env_value("ZERODHA_API_SECRET", api_secret.strip())
    if access_token is not None:
        _save_env_value("ZERODHA_ACCESS_TOKEN", access_token.strip())
    if user_id is not None:
        _save_env_value("ZERODHA_USER_ID", user_id.strip())
    _reload_settings()


def request_token_to_access_token(request_token: str) -> dict[str, Any]:
    if not KITE_API_SECRET:
        raise RuntimeError("Missing ZERODHA_API_SECRET for access token exchange.")
    kite = get_kite_client()
    session = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
    access_token = session.get("access_token")
    if access_token:
        save_zerodha_credentials(access_token=access_token, user_id=str(session.get("user_id", "")))
    return session


def normalize_symbol(symbol: str) -> tuple[str, str]:
    symbol = symbol.strip().upper()
    if ".NS" in symbol:
        return "NSE", symbol.replace(".NS", "")
    if ".BO" in symbol:
        return "BSE", symbol.replace(".BO", "")
    if ":" in symbol:
        exchange, tradingsymbol = symbol.split(":", 1)
        return exchange.upper(), tradingsymbol
    return "NSE", symbol


def get_quote(symbol: str) -> dict[str, Any]:
    kite = get_kite_client()
    exchange, tradingsymbol = normalize_symbol(symbol)
    quote_key = f"{exchange}:{tradingsymbol}"
    return kite.ltp(quote_key)


def place_zerodha_order(symbol: str,
                        transaction_type: str,
                        quantity: int,
                        order_type: str = "MARKET",
                        product: str = "MIS",
                        price: float | None = None,
                        trigger_price: float | None = None,
                        variety: str = "regular") -> dict[str, Any]:
    kite = get_kite_client()
    exchange, tradingsymbol = normalize_symbol(symbol)
    if quantity <= 0:
        raise ValueError("Quantity must be greater than 0.")

    order = {
        "variety": variety,
        "exchange": exchange,
        "tradingsymbol": tradingsymbol,
        "transaction_type": transaction_type.upper(),
        "quantity": quantity,
        "product": product.upper(),
        "order_type": order_type.upper(),
    }
    if order_type.upper() != "MARKET":
        if price is None or price <= 0:
            raise ValueError("Limit orders require a valid price.")
        order["price"] = price
    if trigger_price is not None and trigger_price > 0:
        order["trigger_price"] = trigger_price
    return kite.place_order(**order)


def get_positions() -> list[dict[str, Any]]:
    kite = get_kite_client()
    data = kite.positions()
    return data.get("net", []) if isinstance(data, dict) else []


def get_holdings() -> list[dict[str, Any]]:
    kite = get_kite_client()
    data = kite.holdings()
    return data if isinstance(data, list) else []


def get_orders() -> list[dict[str, Any]]:
    kite = get_kite_client()
    data = kite.orders()
    return data if isinstance(data, list) else []
