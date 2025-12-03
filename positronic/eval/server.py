import configuronic as cfn

from positronic.server import positronic_server as ps
from positronic.utils import s3 as pos3

from .ds import evaluation_ds

config = ps.main.override(
    dataset=evaluation_ds,
    episode_keys={
        'duration': {'label': 'Duration', 'format': '%.2f sec'},
        'task_code': {
            'label': 'Task',
            'filter': True,
            'renderer': {
                'type': 'badge',
                'options': {
                    'towels': {'label': 'Towels', 'variant': 'danger'},
                    'spoons': {'label': 'Spoons', 'variant': 'warning'},
                    'scisors': {'label': 'Scisors', 'variant': 'success'},
                },
            },
        },
        'inference.policy.name': {
            'label': 'Policy',
            'filter': True,
            'renderer': {
                'type': 'badge',
                'options': {
                    'act': {'label': 'ACT', 'variant': 'danger'},
                    'gr00t': {'label': 'Groot', 'variant': 'warning'},
                    'openpi 0.5': {'label': 'OpenPI 0.5', 'variant': 'success'},
                },
            },
        },
        'eval.external_camera': {
            'label': 'External camera',
            'filter': True,
            'renderer': {
                'type': 'badge',
                'options': {
                    'left': {'label': 'Left', 'variant': 'danger'},
                    'right': {'label': 'Right', 'variant': 'warning'},
                    # NA Should have no badge
                },
            },
        },
        'eval.tote_placement': {
            'label': 'Source tote placement',
            'filter': True,
            'renderer': {
                'type': 'badge',
                'options': {
                    'left': {'label': 'Left', 'variant': 'danger'},
                    'right': {'label': 'Right', 'variant': 'warning'},
                    # NA Should have no badge
                },
            },
        },
        'full_success': {
            'label': 'Success',
            'filter': True,
            'renderer': {
                'type': 'badge',
                'options': {
                    True: {'label': 'Success', 'variant': 'success'},
                    False: {'label': 'Failure', 'variant': 'danger'},
                },
            },
        },
    },
)


if __name__ == '__main__':
    with pos3.mirror():
        cfn.cli(config)
