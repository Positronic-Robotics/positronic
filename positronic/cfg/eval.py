from datetime import datetime

import positronic.cfg.dataset as base_cfg
from positronic.dataset.episode import Episode
from positronic.dataset.transforms.episode import Derive, Identity


def task_code(ep: Episode) -> str:
    match ep['task']:
        case 'Pick all the towels one by one from transparent tote and place them into the large grey tote.':
            return 'Towels'
        case 'Pick all the wooden spoons one by one from transparent tote and place them into the large grey tote.':
            return 'Wooden spoons'
        case 'Pick all the scissors one by one from transparent tote and place them into the large grey tote.':
            return 'Scissors'


def model(ep: Episode) -> str:
    match ep['inference.policy.name']:
        case 'act':
            return 'Action Chunking Trasnformer'
        case 'groot':
            return 'Nvidia Gr00t'
        case 'openpi':
            return 'Open PI 0.5'


def units(ep: Episode) -> str:
    return f'{ep["eval.successful_items"]}/{ep["eval.total_items"]}'


def full_success(ep: Episode) -> bool:
    return (ep['eval.successful_items'] == ep['eval.total_items']) and not ep['eval.aborted']


def uph(ep: Episode) -> float | None:
    items = ep['eval.successful_items']
    if items == 0:
        return None
    return ep.duration_ns / 1e9 / items


ds = base_cfg.transform.override(
    base=base_cfg.local,
    transforms=[
        base_cfg.group.override(
            transforms=[
                Identity(),
                Derive(
                    task_code=task_code,
                    model=model,
                    units=units,
                    full_success=full_success,
                    uph=uph,
                    success=lambda ep: 100 * ep['eval.successful_items'] / ep['eval.total_items'],
                    started=lambda ep: datetime.fromtimestamp(ep.meta['created_ts_ns'] / 1e9),
                ),
            ]
        )
    ],
)
