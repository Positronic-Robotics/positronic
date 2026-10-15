from utilities.plot_eval import success_rate_points


def test_the_success_rate_counts_only_scored_episodes_and_skips_a_checkpoint_with_none():
    data_by_ckpt = {'1000': [{'success': True}, {'success': False}, {'success': None}], '2000': [{'success': None}]}

    assert success_rate_points(data_by_ckpt, ['1000', '2000']) == (['1000'], [50.0])
