"""Cross-tenant authentication for the Dataverse MCP server using device code flow.

Uses MSAL to authenticate a user account in a target tenant via device code,
then acquires a Dataverse-scoped access token. Tokens are cached to disk so
you only need to sign in once (until the refresh token expires).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import msal

logger = logging.getLogger(__name__)

# Well-known Dynamics 365 / Power Platform public client ID (pre-consented for Dataverse)
_DEFAULT_CLIENT_ID = "51f81489-12ee-4a9e-aaae-a2591f45987d"
_CACHE_FILE = Path(__file__).parent.parent.parent / ".mcp_token_cache.bin"
_AUTHORITY_TEMPLATE = "https://login.microsoftonline.com/{tenant_id}"


def get_dataverse_token(
    org_url: str,
    tenant_id: str | None = None,
    client_id: str = _DEFAULT_CLIENT_ID,
) -> str:
    """Acquire a Dataverse access token via device code flow with caching.

    Args:
        org_url: Dataverse org URL (e.g. https://myorg.crm.dynamics.com)
        tenant_id: Target tenant ID. If None, uses "organizations" (multi-tenant).
        client_id: App registration client ID for device code flow.

    Returns:
        A valid access token string.
    """
    return _get_token_from_app(*_build_msal_app(org_url, tenant_id, client_id))


def create_token_provider(
    org_url: str,
    tenant_id: str | None = None,
    client_id: str = _DEFAULT_CLIENT_ID,
) -> callable:
    """Return a callable that always provides a fresh access token.

    The callable uses MSAL's silent token acquisition (refresh token) so
    the token is automatically refreshed when it expires during long runs.
    """
    app, scope, cache = _build_msal_app(org_url, tenant_id, client_id)
    # Do the initial auth (may prompt device code if no cache)
    _get_token_from_app(app, scope, cache)

    def _provider() -> str:
        accounts = app.get_accounts()
        if not accounts:
            raise RuntimeError("No cached account — re-run to authenticate")
        result = app.acquire_token_silent([scope], account=accounts[0])
        if result and "access_token" in result:
            _save_cache(cache)
            return result["access_token"]
        raise RuntimeError("Token refresh failed — re-run to re-authenticate")

    return _provider


def _build_msal_app(
    org_url: str,
    tenant_id: str | None = None,
    client_id: str = _DEFAULT_CLIENT_ID,
) -> tuple:
    """Build MSAL app, scope, and cache."""
    authority = _AUTHORITY_TEMPLATE.format(
        tenant_id=tenant_id or "organizations"
    )
    scope = f"{org_url.rstrip('/')}/.default"

    cache = msal.SerializableTokenCache()
    if _CACHE_FILE.exists():
        cache.deserialize(_CACHE_FILE.read_text(encoding="utf-8"))

    app = msal.PublicClientApplication(
        client_id=client_id,
        authority=authority,
        token_cache=cache,
    )
    return app, scope, cache


def _get_token_from_app(app, scope: str, cache) -> str:
    """Acquire token from MSAL app (silent or device code)."""
    accounts = app.get_accounts()
    result = None
    if accounts:
        logger.info("Found cached account: %s", accounts[0].get("username", "unknown"))
        result = app.acquire_token_silent([scope], account=accounts[0])

    if result and "access_token" in result:
        logger.info("Using cached token for %s", accounts[0].get("username", ""))
        _save_cache(cache)
        return result["access_token"]

    # No cached token — start device code flow
    flow = app.initiate_device_flow(scopes=[scope])
    if "user_code" not in flow:
        raise RuntimeError(f"Device code flow failed: {flow.get('error_description', flow)}")

    print(f"\n{'='*60}")
    print(f"  🔑 Sign in to the target tenant")
    print(f"  {flow['message']}")
    print(f"{'='*60}\n")

    result = app.acquire_token_by_device_flow(flow)

    if "access_token" not in result:
        error = result.get("error_description", result.get("error", "Unknown error"))
        raise RuntimeError(f"Authentication failed: {error}")

    logger.info("Authenticated as %s", result.get("id_token_claims", {}).get("preferred_username", ""))
    _save_cache(cache)
    return result["access_token"]


def _save_cache(cache: msal.SerializableTokenCache) -> None:
    """Persist token cache if it changed."""
    if cache.has_state_changed:
        _CACHE_FILE.write_text(cache.serialize(), encoding="utf-8")
        logger.debug("Token cache saved to %s", _CACHE_FILE)
