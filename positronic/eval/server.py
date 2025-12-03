import configuronic as cfn

from positronic.server import positronic_server as ps
from positronic.utils import s3 as pos3

from .ds import evaluation_ds

config = ps.main.override(
    dataset=evaluation_ds,
    episode_keys={
        # TODO: We need to have an ability to remove the first ID column
        'task_code': {'label': 'Task', 'filter': True},
        'model': {'label': 'Model', 'filter': True},
        'units': {'label': 'Units'},
        'uph': {'label': 'UPH', 'format': '%.1f'},
        'success': {'label': 'Success', 'format': '%.1f%%'},
        # 'started': {'label': 'Started', 'format': '%Y-%m-%d %H:%M:%S'},  # This does not work now
        'full_success': {
            'label': 'Status',
            'filter': True,  # Currently filter shows true / false, I expect it to show Pass / Fail
            'renderer': {
                'type': 'badge',
                'options': {
                    True: {'label': 'Pass', 'variant': 'success'},
                    False: {'label': 'Fail', 'variant': 'danger'},
                },
            },
        },
        'duration': {'label': 'Duration', 'format': '%.1f sec'},
    },
)


if __name__ == '__main__':
    with pos3.mirror():
        cfn.cli(config)
