#!/usr/bin/env python3
# deriv_auth_test.py
# Try both authorize shapes against the installed python_deriv_api

import asyncio
import os
from pprint import pprint

async def try_shapes(token):
    try:
        from deriv_api import DerivAPI
    except Exception as e:
        print("deriv_api import FAILED:", e)
        return

    # create api instance (use enforced app id)
    api = DerivAPI(app_id=71710)
    # Try raw string authorize
    try:
        print("Trying api.authorize(token_str)...")
        resp = await api.authorize(token)
        print("RAW style success (api.authorize(token)):")
        pprint(resp)
    except Exception as e:
        print("RAW style failed:", repr(e))

    # Try dict authorize
    try:
        print("\nTrying api.authorize({'authorize': token})...")
        resp2 = await api.authorize({"authorize": token})
        print("DICT style success (api.authorize({'authorize': ...})): ")
        pprint(resp2)
    except Exception as e:
        print("DICT style failed:", repr(e))

    try:
        await api.close()
    except Exception:
        pass

if __name__ == "__main__":
    token = os.environ.get("DERIV_TEST_TOKEN") or input("Token: ").strip()
    asyncio.run(try_shapes(token))