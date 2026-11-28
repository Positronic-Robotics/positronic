"""The rig of the DAgger rounds on an env server: a simulator in a process of its own, which ``RemoteEnvControlSystem``
drives over a socket."""

from collections.abc import Callable, Collection, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import cached_property
from typing import Any, Protocol

from eval_vocabulary.episode import GT_STATE_SIGNAL

import pimm
from positronic import wire
from positronic.dagger.rounds import Rig, run_harness
from positronic.eval import Embodiment
from positronic.policy.base import Policy
from positronic.policy.harness import Harness, Rollout
from positronic.simulator.env_server.adapter import EnvAdapter
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem


class MakeTeleoperator(Protocol):
    """Makes the teleoperator of a Collect step. With ``takes_over_at_start``, the teleoperator takes over at the start
    of each episode."""

    def __call__(self, *, takes_over_at_start: bool) -> Policy: ...


class EnvServerRig(Rig):
    """A simulated rig: an env server that ``serve`` runs for each Collect step, through ``adapter``. ``embodiment``
    makes the embodiment of the proxy of the env server, as ``remote_embodiment`` does. ``teleoperator`` makes the
    teleoperator of each Collect step. The helper of a run gets the privileged signal ``ground_truth`` of the proxy as
    its ground truth."""

    def __init__(
        self,
        adapter: EnvAdapter,
        serve: Callable[[], AbstractContextManager[tuple[str, int]]],
        *,
        embodiment: Callable[[RemoteEnvControlSystem], Embodiment],
        ground_truth: str,
        teleoperator: MakeTeleoperator,
    ):
        self._adapter = adapter
        self._serve = serve
        self._embodiment = embodiment
        self._ground_truth = ground_truth
        self._teleoperator = teleoperator

    @cached_property
    def _unserved(self) -> Embodiment:
        # The proxy connects at its first request, so a proxy that the rig does not run starts no env server.
        return self._embodiment(RemoteEnvControlSystem(self._adapter, nullcontext(('localhost', 0))))

    @property
    def observations(self) -> Collection[str]:
        return self._unserved.observations.keys()

    @property
    def commands(self) -> Collection[str]:
        return self._unserved.commands.keys()

    @contextmanager
    def harness(self, *, takes_over_at_start: bool) -> Iterator[tuple[Callable[[Rollout], dict[str, Any]], Policy]]:
        with pimm.World(virtual_time=True) as world:
            proxy = RemoteEnvControlSystem(self._adapter, self._serve())
            embodiment = self._embodiment(proxy)
            harness = Harness(embodiment)
            privileged = {GT_STATE_SIGNAL: proxy.privileged[self._ground_truth]}
            wire.wire_embodiment(world, harness, embodiment, privileged=privileged, done=proxy.done)
            with run_harness(world, harness, embodiment.control_systems) as perform:
                yield perform, self._teleoperator(takes_over_at_start=takes_over_at_start)
