import asyncio
import time


class RateLimitExceeded(Exception):
    pass


class AsyncTokenBucket:
    def __init__(self, rate: float, capacity: int):
        self._rate = rate          # tokens per second
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                wait = (1.0 - self._tokens) / self._rate

            if time.monotonic() + wait > deadline:
                raise RateLimitExceeded("Rate limit timeout exceeded")
            await asyncio.sleep(wait)
