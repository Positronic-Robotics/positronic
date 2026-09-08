"""What the platform offers: the evals a caller may name, and the tasks a caller may compose into one.

Both lists are answered for the caller's grant. Every registered user sees the evals a submission names.
A caller with a customer grant also sees the rig's evals and tasks, filtered to the entries offered
to the grant's client, and only that caller may file a plan over them.
"""

from __future__ import annotations

from platform_client.enums import CameraVantage, Placement
from platform_client.evals import EvalRef
from platform_client.slug import Slugged
from platform_client.tasks import TaskRef
from pydantic import BaseModel, Field


class TaskSummary(BaseModel):
    """`catalog.tasks` — one task the caller may put in a plan, as the catalogue states it.

    `tote_placement` and `external_cameras` list the sides a plan may pin, or draw between; a mount
    is keyed by the name the task gives it. `default_cap_per_episode_sec` is what a plan that states
    no cap takes.
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


class EvalSummary(BaseModel):
    """`catalog.evals` — one eval `evals.run` accepts by name.

    `tasks` names what the eval runs, in the embodiment's own spelling. A composable eval is a plan
    the registry holds over catalogue tasks, so a caller may write their own over the same tasks; a
    pinned one fixes its trials, and takes no `tasks` of the caller's.
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
