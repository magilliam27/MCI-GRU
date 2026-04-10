# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "colab-mcp @ git+https://github.com/googlecolab/colab-mcp",
# ]
# ///
"""Wrapper to fix fakeredis compatibility before launching colab-mcp.

fakeredis >= 2.31.1 renamed FakeConnection to FakeRedisConnection,
but docket/_redis.py still imports the old name. This wrapper patches
the module before the server starts.
"""
import fakeredis.aioredis as _aioredis

if not hasattr(_aioredis, "FakeConnection"):
    for candidate in ("FakeRedisConnection", "FakeAsyncRedisConnection"):
        if hasattr(_aioredis, candidate):
            _aioredis.FakeConnection = getattr(_aioredis, candidate)
            break

from colab_mcp import main

main()
