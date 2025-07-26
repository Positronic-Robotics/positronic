import configuronic as cfn
import positronic.cfg.inference.action
import positronic.cfg.inference.policy
import positronic.cfg.inference.state
from positronic.inference.inference import Inference

umi_inference = cfn.Config(
    Inference,
    state_encoder=positronic.cfg.inference.state.end_effector,
    policy=positronic.cfg.inference.policy.act,
    action_decoder=positronic.cfg.inference.action.umi_relative,
    rerun=True,
    device='cuda'
)
