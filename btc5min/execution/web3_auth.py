"""
py-clob-client bootstrapper — builds a single `ClobClient` per process.

Replaces the previous `simmer_sdk.SimmerClient` layer. We now talk
directly to Polymarket's L2 CLOB (Polygon) via the official Polymarket
`py-clob-client` Python client, which handles EIP-712 signing and API
credential derivation internally.

Why we ripped out Simmer:
- Simmer is a Market Maker router that required `client.import_market()`
  for every market the bot touched, capped at 10 imports/day. Completely
  incompatible with our Sniper HFT that trades "Flash Markets" (5-minute
  BTC up/down windows) generated on the fly.
- Going direct to the CLOB means no daily market-import limit, no extra
  hop, and native support for all Polymarket markets.

Delegated proxy wallets (Google/Email login):
- A Polymarket account created via Google/Email login uses a Safe/Proxy
  contract on Polygon that holds the USDC.e balance. The signing EOA
  (private key) is different from the on-chain funder address. This is
  "signature_type=1" (POLY_PROXY) in py-clob-client parlance — we must
  pass BOTH the signing key AND the `funder` (proxy address).
- The user is expected to set:
    WALLET_PRIVATE_KEY       → the L2 EOA signing key
    POLYMARKET_PROXY_FUNDER  → the public 0x... proxy/Safe address

The private key is never logged. We only log the public signing address
(derived via eth_account) and the funder address for operator confidence.
"""

import threading

from ..config import (
    TRADING_MODE, ETH_PRIVATE_KEY, POLYMARKET_PROXY_FUNDER,
    POLYMARKET_CLOB_URL, CHAIN_ID_POLYGON, log,
)

# Process-wide singleton — derive_api_creds does an HTTP handshake.
_client_lock = threading.Lock()
_client_singleton = None

# Signature type constants (per py-clob-client / Polymarket docs):
#   0 = EOA (plain MetaMask / exported seed phrase wallet)
#   1 = POLY_PROXY (Email/Google login → Safe-style proxy contract)
#   2 = POLY_GNOSIS_SAFE (Legacy MagicLink)
# Como el usuario ya apuntó al Funder correcto de su cuenta real, restauramos al Tipo 1 (Proxy)
SIGNATURE_TYPE_POLY_PROXY = 1


def get_clob_host() -> str:
    """Return the CLOB venue URL for the current TRADING_MODE."""
    if TRADING_MODE == "live_mainnet":
        return POLYMARKET_CLOB_URL
    # paper_simmer legacy label — no longer a real venue. Only live_mainnet
    # or offline_sim are valid. Return empty to force fallback.
    return ""


def get_clob_client():
    """Return the process-wide `py_clob_client.ClobClient`, or None.

    Returns None (and logs a warning) when:
    - TRADING_MODE is offline_sim (or legacy paper_simmer)
    - py-clob-client is not installed
    - WALLET_PRIVATE_KEY or POLYMARKET_PROXY_FUNDER is missing/invalid
    - client construction or API credential derivation raises

    The first successful call performs:
        client = ClobClient(host, key, chain_id, signature_type=1, funder=...)
        client.set_api_creds(client.create_or_derive_api_creds())

    which is idempotent server-side — derived API creds are deterministic
    from the signing key.
    """
    global _client_singleton

    if TRADING_MODE in ("offline_sim", "paper_simmer"):
        return None

    with _client_lock:
        if _client_singleton is not None:
            return _client_singleton

        # ── Validate creds ────────────────────────────────────
        pk = ETH_PRIVATE_KEY
        if not pk or len(pk) < 60:
            log.warning(
                "[CLOB] WALLET_PRIVATE_KEY missing or invalid length — "
                "cannot init py-clob-client (need the L2 signing key)."
            )
            return None

        funder = (POLYMARKET_PROXY_FUNDER or "").strip()
        if not funder or not funder.startswith("0x") or len(funder) != 42:
            log.warning(
                "[CLOB] POLYMARKET_PROXY_FUNDER not set or invalid — "
                "delegated Polymarket wallets REQUIRE the public proxy "
                "address (0x..., 42 chars) to route orders to the right "
                "USDC.e balance. Falling back to offline_sim."
            )
            return None

        # ── Import the SDK lazily ─────────────────────────────
        try:
            from py_clob_client.client import ClobClient  # type: ignore
        except ImportError as e:
            log.error(
                f"[CLOB] Error crítico importando py-clob-client (Falló una DLL o dependencia): {e}"
            )
            return None

        # ── Construct client (signature_type=1 POLY_PROXY) ────
        host = POLYMARKET_CLOB_URL
        normalized_key = pk if pk.startswith("0x") else "0x" + pk
        try:
            client = ClobClient(
                host,
                key=normalized_key,
                chain_id=CHAIN_ID_POLYGON,
                signature_type=SIGNATURE_TYPE_POLY_PROXY,
                funder=funder,
            )
        except Exception as e:
            log.error(f"[CLOB] ClobClient construction failed: {e}")
            return None

        # ── Derive / load L2 API credentials ──────────────────
        # create_or_derive_api_creds() is the canonical one-shot: the
        # CLOB hashes the signing key deterministically so repeated calls
        # return the same (api_key, secret, passphrase) triple.
        try:
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
        except Exception as e:
            log.error(
                f"[CLOB] create_or_derive_api_creds failed: {e} — "
                f"check that signing key / funder match a real Polymarket account"
            )
            return None

        _client_singleton = client

        # ── Logging: public addrs only, never the private key ─
        try:
            from eth_account import Account
            signing_addr = Account.from_key(normalized_key).address
        except Exception:
            signing_addr = "<unknown>"

        log.debug(
            f"[CLOB] py-clob-client ready | host={host} | "
            f"chain_id={CHAIN_ID_POLYGON} | signature_type=POLY_PROXY(1) | "
            f"signer={signing_addr} | funder={funder}"
        )
        return _client_singleton
