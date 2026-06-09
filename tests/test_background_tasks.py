"""Tests for fire-and-forget background task retention.

`asyncio` keeps only a weak reference to a task, so a bare
`asyncio.create_task(...)` whose handle is discarded can be garbage-collected
mid-execution. `main._spawn` retains a strong reference until the task finishes.
"""
import asyncio
import weakref

import pytest

from backend import main


@pytest.mark.asyncio
async def test_spawn_retains_task_until_done():
    main._background_tasks.clear()
    started = asyncio.Event()

    async def work():
        started.set()
        await asyncio.sleep(0.01)

    task = main._spawn(work())
    assert task in main._background_tasks          # strong ref held while pending
    await started.wait()
    await task
    await asyncio.sleep(0)                          # let the done-callback run
    assert task not in main._background_tasks       # cleaned up after completion


@pytest.mark.asyncio
async def test_spawn_keeps_strong_reference_after_local_dropped():
    main._background_tasks.clear()

    async def work():
        await asyncio.sleep(0.01)

    ref = weakref.ref(main._spawn(work()))          # don't keep our own handle
    assert ref() is not None                        # survives: the set holds it
    assert ref() in main._background_tasks
    await asyncio.gather(*main._background_tasks)


@pytest.mark.asyncio
async def test_spawn_discards_task_even_on_exception():
    main._background_tasks.clear()

    async def boom():
        raise RuntimeError("boom")

    task = main._spawn(boom())
    with pytest.raises(RuntimeError):
        await task
    await asyncio.sleep(0)
    assert task not in main._background_tasks       # callback still cleans up
