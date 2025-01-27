#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import Coroutine, Optional, Set

from loguru import logger

_TASKS: Set[asyncio.Task] = set()


def create_task(loop: asyncio.AbstractEventLoop, coroutine: Coroutine, name: str) -> asyncio.Task:
    """
    Creates and schedules a new asyncio Task that runs the given coroutine.

    The task is added to a global set of created tasks.

    Args:
        loop (asyncio.AbstractEventLoop): The event loop to use for creating the task.
        coroutine (Coroutine): The coroutine to be executed within the task.
        name (str): The name to assign to the task for identification.

    Returns:
        asyncio.Task: The created task object.
    """

    async def run_coroutine():
        try:
            await coroutine
        except asyncio.CancelledError:
            logger.trace(f"{name}: task cancelled")
            # Re-raise the exception to ensure the task is cancelled.
            raise
        except Exception as e:
            logger.exception(f"{name}: unexpected exception: {e}")

    task = loop.create_task(run_coroutine())
    task.set_name(name)
    _TASKS.add(task)
    logger.trace(f"{name}: task created")
    return task


async def wait_for_task(task: asyncio.Task, timeout: Optional[float] = None):
    """Wait for an asyncio.Task to complete with optional timeout handling.

    This function awaits the specified asyncio.Task and handles scenarios for
    timeouts, cancellations, and other exceptions. It also ensures that the task
    is removed from the set of registered tasks upon completion or failure.

    Args:
        task (asyncio.Task): The asyncio Task to wait for.
        timeout (Optional[float], optional): The maximum number of seconds
            to wait for the task to complete. If None, waits indefinitely.
            Defaults to None.
    """
    name = task.get_name()
    try:
        if timeout:
            await asyncio.wait_for(task, timeout=timeout)
        else:
            await task
    except asyncio.TimeoutError:
        logger.warning(f"{name}: timed out waiting for task to finish")
    except asyncio.CancelledError:
        logger.trace(f"{name}: unexpected task cancellation (maybe Ctrl-C?)")
    except Exception as e:
        logger.exception(f"{name}: unexpected exception while stopping task: {e}")
    finally:
        try:
            _TASKS.remove(task)
        except KeyError as e:
            logger.trace(f"{name}: unable to remove task (already removed?): {e}")


async def cancel_task(task: asyncio.Task, timeout: Optional[float] = None):
    """Cancels the given asyncio Task and awaits its completion with an
    optional timeout.

    This function removes the task from the set of registered tasks upon
    completion or failure.

    Args:
        task (asyncio.Task): The task to be cancelled.
        timeout (Optional[float]): The optional timeout in seconds to wait for the task to cancel.

    """
    name = task.get_name()
    task.cancel()
    try:
        if timeout:
            await asyncio.wait_for(task, timeout=timeout)
        else:
            await task
    except asyncio.TimeoutError:
        logger.warning(f"{name}: timed out waiting for task to cancel")
    except asyncio.CancelledError:
        # Here are sure the task is cancelled properly.
        pass
    except Exception as e:
        logger.exception(f"{name}: unexpected exception while cancelling task: {e}")
    finally:
        try:
            _TASKS.remove(task)
        except KeyError as e:
            logger.trace(f"{name}: unable to remove task (already removed?): {e}")


def current_tasks() -> Set[asyncio.Task]:
    """Returns the list of currently created/registered tasks."""
    return _TASKS