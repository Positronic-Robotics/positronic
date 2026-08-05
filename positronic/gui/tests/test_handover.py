from datetime import datetime, timedelta

import pytest
import yaml

from positronic.gui import handover


def write_assignment(directory, **fields):
    payload = {
        'schema_version': handover.SCHEMA_VERSION,
        'batch_id': '2026-08-05-a',
        'task': 'Pick up objects from the red tote and place them in the green tote.',
        'episode_target': 20,
        'created_at': '2026-08-05T09:14:03+00:00',
        **fields,
    }
    (directory / handover.ASSIGNMENT_NAME).write_text(yaml.safe_dump(payload))
    return payload


def test_read_assignment_is_none_without_a_file(tmp_path):
    assert handover.read_assignment(tmp_path) is None


def test_read_assignment_is_none_when_the_directory_is_absent(tmp_path):
    assert handover.read_assignment(tmp_path / 'nowhere') is None


def test_read_assignment_reads_every_field(tmp_path):
    write_assignment(tmp_path, notes='tote on the left')

    assignment = handover.read_assignment(tmp_path)
    assert assignment is not None

    assert assignment == handover.Assignment(
        schema_version=1,
        batch_id='2026-08-05-a',
        task='Pick up objects from the red tote and place them in the green tote.',
        episode_target=20,
        notes='tote on the left',
        created_at='2026-08-05T09:14:03+00:00',
    )
    assert assignment.supported


def test_notes_are_optional(tmp_path):
    write_assignment(tmp_path)

    assignment = handover.read_assignment(tmp_path)
    assert assignment is not None

    assert assignment.notes == ''


def test_a_newer_schema_version_reports_only_its_version(tmp_path):
    write_assignment(tmp_path, schema_version=handover.SCHEMA_VERSION + 1)

    assignment = handover.read_assignment(tmp_path)
    assert assignment is not None

    assert not assignment.supported
    assert assignment.schema_version == handover.SCHEMA_VERSION + 1
    assert assignment.task == ''


def test_a_missing_required_field_surfaces(tmp_path):
    payload = write_assignment(tmp_path)
    del payload['episode_target']
    (tmp_path / handover.ASSIGNMENT_NAME).write_text(yaml.safe_dump(payload))

    with pytest.raises(KeyError):
        handover.read_assignment(tmp_path)


def test_an_existing_marker_marks_the_batch_handed_back(tmp_path):
    assert not handover.completion_path(tmp_path, '2026-08-05-a').exists()

    handover.write_completion(tmp_path, '2026-08-05-a')

    assert handover.completion_path(tmp_path, '2026-08-05-a').exists()


def test_write_completion_names_the_marker_after_the_batch(tmp_path):
    path = handover.write_completion(tmp_path, '2026-08-05-a')

    assert path == tmp_path / 'done-2026-08-05-a.yaml'
    assert path == handover.completion_path(tmp_path, '2026-08-05-a')


def test_completion_marker_carries_the_batch_and_a_utc_timestamp(tmp_path):
    written = yaml.safe_load(handover.write_completion(tmp_path, '2026-08-05-a').read_text())

    assert written['batch_id'] == '2026-08-05-a'
    assert datetime.fromisoformat(written['finished_at']).utcoffset() == timedelta(0)
    assert 'operator_note' not in written


def test_an_operator_note_is_recorded_when_given(tmp_path):
    written = yaml.safe_load(handover.write_completion(tmp_path, 'b', 'gripper slipped twice').read_text())

    assert written['operator_note'] == 'gripper slipped twice'
