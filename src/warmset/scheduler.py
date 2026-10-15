"""Ordering of model calls.

On a machine that holds one model at a time, the order of calls decides how many reloads happen.
``ModelScheduler`` grants model calls up to ``capacity`` at a time. In FIFO mode it is a plain
semaphore. In residency mode it keeps serving the model that is loaded while anyone is waiting for
it, and switches only when that queue is empty: with a batch of independent tasks in flight, all
small-model work runs, then the big model is loaded once for every escalation.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class ModelScheduler:
    def __init__(self, capacity: int = 1, *, residency_first: bool = False, switch_grace: float = 0.02,
                 current: str | None = None) -> None:
        self.capacity = capacity
        self.residency_first = residency_first
        self.switch_grace = switch_grace  # let just-finished tasks re-queue before switching models
        self.current = current
        self.switches = 0
        self._busy = 0
        self._waiting: list[tuple[str, asyncio.Future[None]]] = []
        self._dispatch_pending = False

    @asynccontextmanager
    async def slot(self, model: str) -> AsyncIterator[None]:
        await self._acquire(model)
        try:
            yield
        finally:
            self._release()

    def _can_take_now(self, model: str) -> bool:
        if self._busy >= self.capacity or self._waiting or self._dispatch_pending:
            return False
        return not self.residency_first or self._busy == 0 or self.current in (None, model)

    async def _acquire(self, model: str) -> None:
        if self._can_take_now(model):
            self._grant(model)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiting.append((model, fut))
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self._release()  # granted just as we were cancelled: give the slot back
            else:
                self._waiting = [(m, f) for m, f in self._waiting if f is not fut]
            raise

    def _grant(self, model: str) -> None:
        if self.current is not None and model != self.current:
            self.switches += 1
        self.current = model
        self._busy += 1

    def _release(self) -> None:
        self._busy -= 1
        if not self._waiting:
            return
        if self.residency_first and self.switch_grace > 0:
            if not self._dispatch_pending:
                self._dispatch_pending = True
                asyncio.get_running_loop().call_later(self.switch_grace, self._dispatch)
        else:
            self._dispatch()

    def _dispatch(self) -> None:
        self._dispatch_pending = False
        self._waiting = [(m, f) for m, f in self._waiting if not f.cancelled()]
        while self._busy < self.capacity and self._waiting:
            idx = 0
            if self.residency_first:
                same = [i for i, (m, _) in enumerate(self._waiting) if m == self.current]
                if same:
                    idx = same[0]
                elif self._busy > 0:
                    return  # do not switch models while calls on the loaded one are still running
            model, fut = self._waiting.pop(idx)
            self._grant(model)
            fut.set_result(None)
