import pickle

from positronic.teleop.clutch import OperatorPosition


def test_an_operator_position_pickles_by_its_name():
    assert pickle.loads(pickle.dumps(OperatorPosition.BACK)) is OperatorPosition.BACK
