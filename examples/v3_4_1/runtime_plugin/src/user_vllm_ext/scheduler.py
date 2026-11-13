"""User-defined scheduler loaded by the real vLLM engine."""

import logging

from vllm.v1.core.sched.scheduler import Scheduler


logger = logging.getLogger(__name__)


class UserScheduler(Scheduler):
    """First demonstration of a custom scheduling class."""

    def __init__(self, *args, **kwargs):
        logger.warning(
            "V341_USER_SCHEDULER_INSTANTIATED"
        )
        super().__init__(*args, **kwargs)
