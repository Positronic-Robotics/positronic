"""The tasks a RoboLab env server can build: RoboLab's own benchmark and the tasks of each task library it was given.

A task library is a directory outside the RoboLab checkout that holds ``tasks/<file>.py`` (one RoboLab ``Task``
class each, as RoboLab's own task files), ``_metadata/task_metadata.json`` in RoboLab's metadata schema, and the
scenes and assets its tasks load. It may also hold ``cameras.py``, whose ``camera_sets()`` returns camera sets by
name, as RoboLab's ``camera_presets`` holds them.

A library scene holds the robot base, so the env server registers a library task on no robot table fixture
(RoboLab 0.3 and later spawn the fixture of the robot cfg otherwise).

The env server reads this module as a top-level ``task_catalog`` from its own directory, as it reads ``keys``, so it
imports nothing from positronic or Isaac Lab.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METADATA = Path('_metadata') / 'task_metadata.json'


@dataclass(frozen=True)
class CatalogTask:
    name: str
    episode_length_s: float
    # The instruction as the task class states it: one text, or one text per phrasing.
    instruction: str | dict[str, str]
    categories: frozenset[str]


@dataclass(frozen=True)
class LibraryTask:
    task: CatalogTask
    file: Path


def _read(metadata: Path, category_map: dict[str, str]) -> list[tuple[CatalogTask, str]]:
    """Each task that ``metadata`` lists, with the name of its task file."""
    tasks = []
    for entry in json.loads(metadata.read_text()):
        attributes = [a.strip() for a in entry['attributes'].split(',')]
        task = CatalogTask(
            name=entry['task_name'],
            episode_length_s=float(entry['episode_s']),
            # The metadata holds no variants for a task whose instruction is one text.
            instruction=entry['instruction_variants'] or entry['instruction'],
            # An attribute outside the map, such as ``vague``, is a phrasing and no category.
            categories=frozenset(category_map[a] for a in attributes if a in category_map),
        )
        tasks.append((task, entry['filename']))
    return tasks


@dataclass(frozen=True)
class TaskCatalog:
    benchmark: dict[str, CatalogTask]
    library: dict[str, LibraryTask]
    categories: frozenset[str]

    @classmethod
    def load(cls, benchmark_task_dir: Path, task_libs: list[Path], category_map: dict[str, str]) -> 'TaskCatalog':
        benchmark = {task.name: task for task, _ in _read(benchmark_task_dir / METADATA, category_map)}
        library: dict[str, LibraryTask] = {}
        for lib in task_libs:
            for task, filename in _read(lib / METADATA, category_map):
                file = lib / 'tasks' / filename
                # RoboLab names a task's env by its class, so one class name in two places registers one env twice.
                if task.name in benchmark:
                    raise ValueError(
                        f'{file} defines {task.name!r}, the name of the RoboLab benchmark task {task.name!r}'
                    )
                if task.name in library:
                    raise ValueError(f'{file} and {library[task.name].file} both define the task {task.name!r}')
                if not file.is_file():
                    raise FileNotFoundError(f'task library {lib} names {task.name!r} in {file}, which does not exist')
                library[task.name] = LibraryTask(task, file)
        return cls(benchmark, library, frozenset(category_map.values()))

    def select(self, selection: Any) -> list[CatalogTask]:
        """The tasks ``selection`` names: a category, a task name, or a list of names; ``None`` is the whole
        benchmark. A category or ``None`` selects benchmark tasks only, so a library never changes a benchmark
        sweep; a library task runs only by its name."""
        match selection:
            case None:
                names = list(self.benchmark)
            case str() as category if category in self.categories:
                names = [name for name, task in self.benchmark.items() if category in task.categories]
            case str() as name:
                names = [name]
            case _:
                names = list(selection)
        known = {**self.benchmark, **{name: lib_task.task for name, lib_task in self.library.items()}}
        unknown = [name for name in names if name not in known]
        if unknown:
            raise ValueError(
                f'RoboLab does not hold {unknown}; '
                f'a task is a name, a list of names, or one of {sorted(self.categories)}'
            )
        return [known[name] for name in names]

    def register_target(self, name: str) -> str:
        """What RoboLab's registration takes for ``name``: the task file path for a library task (RoboLab then
        names the env by the class in it), the name itself for a benchmark task."""
        lib_task = self.library.get(name)
        return name if lib_task is None else str(lib_task.file)
