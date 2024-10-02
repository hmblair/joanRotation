
from __future__ import annotations
import torch


class EpochTimer:
    """
    Helper code to time an epoch of training.
    """

    def __init__(
        self: EpochTimer,
    ) -> None:

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        self.start.record()

    def finish(self: EpochTimer) -> float:
        """
        Finish timing the code.
        """
        self.end.record()
        torch.cuda.synchronize()
        t = self.start.elapsed_time(self.end)
        return t / 1000
