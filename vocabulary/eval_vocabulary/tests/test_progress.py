from eval_vocabulary.progress import LADDER, STAGE_LABELS, Stage, highest_stage

WIRE_CODES = ('floating', 'reaching', 'contact', 'control', 'at-target')


def test_the_wire_codes_are_what_recordings_carry():
    assert tuple(stage.value for stage in Stage) == WIRE_CODES


def test_the_ladder_is_the_declaration_order():
    assert LADDER == tuple(Stage)


def test_every_stage_has_a_label():
    assert set(STAGE_LABELS) == set(Stage)


def test_the_highest_marked_stage_wins_over_the_last_marked_one():
    assert highest_stage(['floating', 'at-target', 'reaching']) is Stage.AT_TARGET


def test_a_code_outside_the_ladder_is_ignored():
    assert highest_stage(['nudged', 'reaching']) is Stage.REACHING


def test_an_episode_that_marked_nothing_on_the_ladder_has_no_stage():
    assert highest_stage([]) is None
    assert highest_stage(['nudged']) is None
