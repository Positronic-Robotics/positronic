from types import SimpleNamespace

from positronic.vendors.droid_finetune import cosmos3_eval


class FakeTorch:
    """The torch flags that decide which kernels run, as the action experiment leaves them."""

    def __init__(self):
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True))
        self.deterministic_algorithms = (False, False)

    def use_deterministic_algorithms(self, mode, *, warn_only=False):
        self.deterministic_algorithms = (mode, warn_only)


def test_two_runs_on_the_same_weights_pick_the_same_kernels():
    environ, torch = {}, FakeTorch()

    cosmos3_eval.make_deterministic(environ, torch)

    assert environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert environ['FLASH_ATTENTION_DETERMINISTIC'] == '1'
    assert torch.deterministic_algorithms == (True, True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_the_experiment_config_cannot_turn_cudnn_autotuning_back_on():
    # The model loader copies trainer.cudnn from the experiment into torch after make_deterministic runs.
    overrides = cosmos3_eval.experiment_overrides('/opt/vae.pth', [])

    assert 'trainer.cudnn.deterministic=True' in overrides
    assert 'trainer.cudnn.benchmark=False' in overrides


def test_an_override_on_the_command_line_wins_over_the_eval_defaults():
    overrides = cosmos3_eval.experiment_overrides('/opt/vae.pth', ['trainer.cudnn.benchmark=True'])

    assert overrides[-1] == 'trainer.cudnn.benchmark=True'


def test_the_server_scores_without_compiled_kernels():
    assert cosmos3_eval.SETUP_UPDATE['use_torch_compile'] is False


def test_the_eval_prompts_in_json_as_the_experiment_trains():
    args = cosmos3_eval.parse_args(['--checkpoint=/ckpt', '--heldout=/tree', '--output=/out.json', '--vae-path=/vae'])

    assert args.format_prompt_as_json is True


def test_plain_text_prompts_stay_available_on_the_command_line():
    args = cosmos3_eval.parse_args([
        '--checkpoint=/ckpt',
        '--heldout=/tree',
        '--output=/out.json',
        '--vae-path=/vae',
        '--no-format-prompt-as-json',
    ])

    assert args.format_prompt_as_json is False
