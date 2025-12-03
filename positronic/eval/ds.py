import positronic.cfg.dataset as base_cfg
from positronic.dataset.episode import Episode
from positronic.dataset.transforms.episode import Derive, Identity


def task_code(ep: Episode) -> str:
    match ep['task']:
        case 'Pick all the towels one by one from transparent tote and place them into the large grey tote.':
            return 'towels'
        case 'Pick all the wooden spoons one by one from transparent tote and place them into the large grey tote.':
            return 'spoons'
        case 'Pick all the scissors one by one from transparent tote and place them into the large grey tote.':
            return 'scissors'


def full_success(ep: Episode) -> bool:
    return (ep['eval.successful_items'] == ep['eval.total_items']) and not ep['eval.aborted']


evaluation_ds = base_cfg.transform.override(
    base=base_cfg.local,
    transforms=[
        base_cfg.group.override(transforms=[Identity(), Derive(task_code=task_code, full_success=full_success)])
    ],
)
