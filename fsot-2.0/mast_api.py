"""
MAST API client helpers
- build_mast_request: construct MashupRequest JSON object
- mast_invoke: perform HTTP POST/GET to MAST invoke endpoint
- mast_cone_search: convenience wrapper for Mast.Caom.Cone

This module intentionally keeps network calls optional; unit tests validate
request construction without performing network requests.
"""
from __future__ import annotations
import json
import requests
from typing import Any, Dict, Optional

MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"


def build_mast_request(service: str, params: Dict[str, Any], fmt: str = "json",
                       pagesize: int = 2000, timeout: int = 30,
                       removenullcolumns: bool = True,
                       cachebreaker: Optional[str] = None) -> Dict[str, Any]:
    """Return a MashupRequest-style dictionary ready to be serialized.

    This does NOT perform any network activity (use mast_invoke for that).
    """
    req = {
        "service": service,
        "params": params,
        "format": fmt,
        "pagesize": pagesize,
        "removenullcolumns": removenullcolumns,
        "timeout": timeout,
    }
    if cachebreaker:
        req["cachebreaker"] = cachebreaker
    return req


def mast_invoke(request_obj: Dict[str, Any], method: str = "POST", allow_retry: bool = True) -> Dict[str, Any]:
    """Invoke the MAST API with the given request object.

    Returns parsed JSON on success, raises requests.HTTPError on HTTP errors.
    Use try/except around this in production code.
    """
    payload = {"request": request_obj}
    if method.upper() == "POST":
        resp = requests.post(MAST_INVOKE_URL, json=payload, timeout=request_obj.get("timeout", 30))
    else:
        # GET requires the request argument to be stringified
        reqstr = json.dumps(request_obj)
        resp = requests.get(MAST_INVOKE_URL, params={"request": reqstr}, timeout=request_obj.get("timeout", 30))

    resp.raise_for_status()
    return resp.json()


def mast_cone_search(ra: float, dec: float, radius: float, pagesize: int = 1000) -> Dict[str, Any]:
    """Convenience builder for a Mast.Caom.Cone query (does not call network).

    To execute, pass the returned object to mast_invoke().
    """
    params = {"ra": ra, "dec": dec, "radius": radius}
    return build_mast_request("Mast.Caom.Cone", params, pagesize=pagesize)


# Example usage (not executed on import)
if __name__ == "__main__":
    example = mast_cone_search(254.28746, -4.09933, 0.2)
    print(json.dumps(example, indent=2))
