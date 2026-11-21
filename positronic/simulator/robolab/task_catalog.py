"""The tasks and camera sets a RoboLab env server can build: RoboLab's own, and those of each task library it was given.

The env server imports this module as a top-level ``task_catalog`` from its own directory, so it imports nothing from
positronic or Isaac Lab. A task library directory holds:

- ``_metadata/task_metadata.json``, in RoboLab's metadata schema;
- ``tasks/<file>.py``, each binding one RoboLab ``Task`` class at module level under its metadata name;
- the scenes and assets its tasks load;
- optionally ``cameras.py``, whose ``camera_sets()`` returns camera sets by a name in ``keys.CAMERA_SETS``. It may
  import the other files of the library by relative imports.
"""

import ast
import importlib.machinery
import importlib.util
import json
import sys
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

METADATA = Path('_metadata') / 'task_metadata.json'
CAMERAS = 'cameras.py'


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


def _module_level_names(file: Path) -> set[str]:
    """The names ``file`` binds at module level: its classes, and the plain names it assigns (``Task = make(...)``)."""
    names = set()
    for node in ast.parse(file.read_text()).body:
        match node:
            case ast.ClassDef(name=name):
                names.add(name)
            case ast.Assign(targets=targets):
                names.update(target.id for target in targets if isinstance(target, ast.Name))
            case ast.AnnAssign(target=ast.Name(id=name)):
                names.add(name)
    return names


@dataclass(frozen=True)
class TaskCatalog:
    benchmark: dict[str, CatalogTask]
    library: dict[str, LibraryTask]
    categories: frozenset[str]

    @classmethod
    def load(cls, benchmark_task_dir: Path, task_libs: list[Path], category_map: dict[str, str]) -> 'TaskCatalog':
        benchmark = {task.name: task for task, _ in _read(benchmark_task_dir / METADATA, category_map)}
        categories = frozenset(category_map.values())
        library: dict[str, LibraryTask] = {}
        for lib in task_libs:
            for task, filename in _read(lib / METADATA, category_map):
                file = lib / 'tasks' / filename
                # RoboLab names a task's env by its class, so one class name in two places registers one env twice.
                if task.name in benchmark:
                    raise ValueError(f'{file} defines {task.name!r}, which is also a task of the RoboLab benchmark')
                if task.name in library:
                    raise ValueError(f'{file} and {library[task.name].file} both define the task {task.name!r}')
                # ``select`` reads a category's name as that category, so a task of that name could never run.
                if task.name in categories:
                    raise ValueError(f'{file} defines {task.name!r}, which is the name of a RoboLab category')
                if not file.is_file():
                    raise FileNotFoundError(f'task library {lib} names {task.name!r} in {file}, which does not exist')
                if task.name not in _module_level_names(file):
                    raise ValueError(f'{file} binds no module-level {task.name!r}, the task name its metadata gives')
                library[task.name] = LibraryTask(task, file)
        return cls(benchmark, library, categories)

    def select(self, selection: str | Sequence[str] | None) -> list[CatalogTask]:
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


def _drop_package(package: str) -> None:
    for name in [n for n in sys.modules if n == package or n.startswith(f'{package}.')]:
        del sys.modules[name]


def _load_cameras_module(lib: Path, package: str) -> ModuleType:
    """``lib``'s ``cameras.py`` as ``<package>.cameras`` of a stand-in package rooted at ``lib``.

    Its relative imports resolve to the files of ``lib``, and ``sys.path`` stays as it is.
    """
    _drop_package(package)
    package_spec = importlib.machinery.ModuleSpec(package, None, is_package=True)
    package_spec.submodule_search_locations = [str(lib)]
    sys.modules[package] = importlib.util.module_from_spec(package_spec)
    spec = importlib.util.spec_from_file_location(f'{package}.cameras', lib / CAMERAS)
    assert spec is not None and spec.loader is not None  # a file that exists always yields a spec
    module = importlib.util.module_from_spec(spec)
    # A dataclass with postponed annotations resolves its module through ``sys.modules`` while it is defined.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        _drop_package(package)
        raise
    return module


def camera_sets(
    task_libs: Sequence[Path], presets: dict[str, list[Any]], selectable: Collection[str]
) -> dict[str, list[Any]]:
    """RoboLab's camera ``presets`` and each set that a task library's ``cameras.py`` returns, by name.

    ``selectable`` names the sets a run can pick (``keys.CAMERA_SETS``). A library set outside it, or one whose name
    RoboLab or another library already gives, raises ``ValueError``.
    """
    sets = dict(presets)
    sources: dict[str, Path | str] = dict.fromkeys(presets, 'RoboLab')
    for index, lib in enumerate(task_libs):
        path = lib / CAMERAS
        if not path.is_file():
            continue
        for name, cameras in _load_cameras_module(lib, f'task_lib_{index}').camera_sets().items():
            if name not in selectable:
                raise ValueError(
                    f'{path} returns the camera set {name!r}, which no run can select: {sorted(selectable)}'
                )
            if name in sets:
                raise ValueError(f'{path} returns the camera set {name!r}, which {sources[name]} already gives')
            sets[name], sources[name] = cameras, path
    return sets
