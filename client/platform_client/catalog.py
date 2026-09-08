"""What the platform offers: the evals a caller may name, and the tasks a caller may compose into one.

Both lists depend on the caller's grant. Every registered user sees the evals a submission
names, and a customer grant adds the rig's evals and tasks.
"""

from __future__ import annotations

from platform_client.enums import CameraVantage, Placement
from platform_client.eval_plan import Clutter
from platform_client.evals import EvalRef
from platform_client.slug import Slugged
from platform_client.tasks import TaskRef
from pydantic import BaseModel, Field


class TaskSummary(BaseModel):
    """`catalog.tasks` — one task the caller may put in a plan, as the catalogue states it.

    `tote_placement` and `external_cameras` list the sides a plan may choose, or ask `random` to
    draw from. `external_cameras` is keyed by the mount name the task defines.
    `default_cap_per_episode_sec` applies when the plan states no cap. `clutter` applies when the
    plan states none; absent, the table is laid bare.
    """

    id: TaskRef
    embodiment: str
    task: str
    setup: str = ''
    total_items: int | None = None
    objects: list[str] = Field(default_factory=list)
    tote_placement: list[Slugged[Placement]] = Field(default_factory=list)
    camera_vantage: Slugged[CameraVantage] | None = None
    external_cameras: dict[str, list[Slugged[Placement]]] = Field(default_factory=dict)
    default_cap_per_episode_sec: int | None = None
    clutter: Clutter | None = None


class EvalSummary(BaseModel):
    """`catalog.evals` — one eval `evals.run` accepts by name.

    `tasks` names what the eval runs, in the embodiment's own spelling. `composable` is true for an
    eval built from catalogue tasks: a caller may compose a plan over the same tasks. A pinned eval
    fixes its trials and accepts no `tasks` from the caller.
    """

    id: EvalRef
    embodiment: str
    tasks: list[str] = Field(default_factory=list)
    composable: bool
    primary_metric: str | None = None
    description: str = ''


class TaskListResponse(BaseModel):
    tasks: list[TaskSummary] = Field(default_factory=list)


class EvalListResponse(BaseModel):
    evals: list[EvalSummary] = Field(default_factory=list)
