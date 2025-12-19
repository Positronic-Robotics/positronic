from datetime import datetime

import configuronic as cfn

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
    match ep['inference.policy.type']:
        case 'act':
            return 'Action Chunking Trasnformer'
        case 'groot':
            return 'Nvidia Gr00t'
        case 'openpi':
            return 'Open PI 0.5'


def ckpt(ep: Episode) -> str | None:
    try:
        match ep['inference.policy.type']:
            case 'act':
                path = ep['inference.policy.checkpoint_path'].split('/checkpoints/', 1)[1]
                # Path to ckpt id: full_ft_q/act/031225/checkpoints/300000/pretrained_model/ -> full_ft_q\031225\300000
                parts = path.split('/')
                if len(parts) >= 5:
                    path = f'{parts[0]}\\{parts[2]}\\{parts[4]}'
                return path
            case 'openpi':
                if 'inference.policy.checkpoint_path' in ep:
                    path = ep['inference.policy.checkpoint_path']
                else:
                    path = ep['inference.policy.server.directory']
                path = path.split('/checkpoints/', 1)[1]
                # Path to ckpt id: full_ft/openpi/pi05_positronic_lowmem/061025/119999 -> full_ft\061025\119999
                parts = path.split('/')
                if len(parts) >= 3:
                    path = f'{parts[0]}\\{parts[-2]}\\{parts[-1]}'
                return path
        return ''
    except Exception:
        return ''


def units(ep: Episode) -> str:
    return f'{ep["eval.successful_items"]}/{ep["eval.total_items"]}'


def uph(ep: Episode) -> float | None:
    items = ep['eval.successful_items']
    if items == 0:
        return None
    return items / (ep.duration_ns / 1e9 / 3600)


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
                    uph=uph,
                    checkpoint=ckpt,
                    success=lambda ep: 100 * ep['eval.successful_items'] / ep['eval.total_items'],
                    started=lambda ep: datetime.fromtimestamp(ep.meta['created_ts_ns'] / 1e9),
                ),
            ]
        )
    ],
)


@cfn.config()
def eval_table():
    return {
        '__index__': {'label': '#', 'format': '%d'},
        'task_code': {'label': 'Task', 'filter': True},
        'model': {'label': 'Model', 'filter': True},
        'checkpoint': {'label': 'Checkpoint'},
        'units': {'label': 'Units'},
        'uph': {'label': 'UPH', 'format': '%.1f'},
        'success': {'label': 'Success', 'format': '%.1f%%'},
        'started': {'label': 'Started', 'format': '%Y-%m-%d %H:%M:%S'},
        'eval.outcome': {
            'label': 'Status',
            'filter': True,
            'renderer': {
                'type': 'badge',
                'options': {
                    # TODO: Currently the filter happens by original data, not the rendered value
                    'Success': {'label': 'Pass', 'variant': 'success'},
                    'Stalled': {'label': 'Fail', 'variant': 'warning'},
                    'Fail': {'label': 'Fail', 'variant': 'warning'},
                    'Ran out of time': {'label': 'Fail', 'variant': 'warning'},
                    'System': {'label': 'Fail', 'variant': 'warning'},
                    'Safety': {'label': 'Safety violation', 'variant': 'danger'},
                },
            },
        },
        '__duration__': {'label': 'Duration', 'format': '%.1f sec'},
    }


LABELS = {'model': 'Model', 'task_code': 'Task', 'checkpoint': 'Checkpoint'}


@cfn.config()
def grouped_table(group_keys: tuple[str, ...] | str):
    if isinstance(group_keys, str):
        group_keys = (group_keys,)

    def group_fn(episodes: list[Episode]):
        duration, suc_items, total_items, assists = 0, 0, 0, 0
        for ep in episodes:
            duration += ep['eval.duration']
            suc_items += ep['eval.successful_items']
            total_items += ep['eval.total_items']
            assists += ep['eval.outcome'] != 'Success'

        result = {key: episodes[0][key] for key in group_keys}
        result.update({
            'UPH': suc_items / (duration / 3600),
            'Success': 100 * suc_items / total_items,
            'MTBF/A': (duration / assists) if assists > 0 else None,
            'Assists': assists,
            'count': len(episodes),
        })
        return result

    format_table = {**{key: {'label': LABELS[key]} for key in group_keys}}
    format_table.update({
        'count': {'label': 'Count'},
        'UPH': {'format': '%.1f'},
        'Success': {'format': '%.2f%%'},
        'MTBF/A': {'format': '%.1f sec', 'default': '-'},
        'Assists': {'format': '%d'},
    })

    group_filter_keys = {key: LABELS[key] for key in group_keys}
    return group_keys, group_fn, format_table, group_filter_keys


# Set of group configurations for evaluation server:
# uv run positronic-server \
#   --dataset=@positronic.cfg.eval.ds \
#   --dataset.base.path /Users/vertix/.cache/positronic/s3/inference/real/191225/ \
#   --port=5001 \
#   --ep_table_cfg=@positronic.cfg.eval.eval_table \
#   --group_tables.models=@positronic.cfg.eval.model_table \
#   --group_tables.model_task=@positronic.cfg.eval.model_task_table \
#   --group_tables.model_ckpt_task=@positronic.cfg.eval.model_chkpt_task_table \
#   --group_tables.model_ckpt=@positronic.cfg.eval.model_chkpt_table

model_table = grouped_table.override(group_keys='model')
model_task_table = grouped_table.override(group_keys=('model', 'task_code'))
model_chkpt_table = grouped_table.override(group_keys=('model', 'checkpoint'))
model_chkpt_task_table = grouped_table.override(group_keys=('model', 'checkpoint', 'task_code'))
