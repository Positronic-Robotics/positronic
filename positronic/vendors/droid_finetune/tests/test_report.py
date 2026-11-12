import pytest

from positronic.vendors.droid_finetune.report import ScoredWindow, build_report, compare, read_report, write_report


def _report(errors: list[float], starts: list[int] | None = None):
    starts = starts or list(range(len(errors)))
    windows = [ScoredWindow(0, start, error) for start, error in zip(starts, errors, strict=True)]
    return build_report(
        windows,
        [0.0] * 8,
        framework='cosmos3',
        checkpoint='ckpt',
        heldout='tree',
        seed=0,
        windows_per_episode=len(errors),
        seconds=1.0,
    )


def test_two_reports_with_equal_window_errors_give_the_same_result():
    result = compare(_report([0.1, 0.2]), _report([0.1, 0.2]), rtol=1e-4)

    assert result.same and result.windows == 2 and result.max_relative_difference == 0


def test_one_window_error_outside_the_tolerance_is_a_different_result():
    assert not compare(_report([0.1, 0.2]), _report([0.1, 0.2002]), rtol=1e-4).same
    assert compare(_report([0.1, 0.2]), _report([0.1, 0.200002]), rtol=1e-4).same


def test_reports_over_different_windows_do_not_compare():
    with pytest.raises(ValueError, match='different windows'):
        compare(_report([0.1, 0.2]), _report([0.1, 0.2], starts=[0, 5]), rtol=1e-4)


def test_a_written_report_reads_back_equal(tmp_path):
    report = _report([0.1, 0.3])
    write_report(report, tmp_path / 'r.json')

    assert read_report(tmp_path / 'r.json') == report
    assert report.summary.action_mse == pytest.approx(0.2)
