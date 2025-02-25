import pytest
import os

import torch
from tools.dataset_dumper import SerialDumper


@pytest.fixture
def sample_data():
    return {"position": torch.tensor([1, 2, 3]), "time": torch.tensor(0.1)}


def test_serial_dumper_empty_dir_first_episode_have_proper_name(
    tmp_path_factory, sample_data
):
    data_dir = tmp_path_factory.mktemp("data")
    dumper = SerialDumper(data_dir)
    dumper.start_episode()
    dumper.write(sample_data)
    dumper.end_episode()

    assert os.path.exists(data_dir / "001.pt")


def test_serial_dumper_episodes_exist_episode_number_increments(
    tmp_path_factory, sample_data
):
    data_dir = tmp_path_factory.mktemp("data")
    with open(data_dir / "001.pt", "wb") as f:
        pass

    dumper = SerialDumper(data_dir)
    dumper.start_episode()
    dumper.write(sample_data)
    dumper.end_episode()

    assert os.path.exists(data_dir / "002.pt")


def test_serial_dumper_data_is_saved_in_episode(tmp_path_factory, sample_data):
    data_dir = tmp_path_factory.mktemp("data")
    dumper = SerialDumper(data_dir)
    dumper.start_episode()
    dumper.write(sample_data)
    dumper.end_episode()

    with open(data_dir / "001.pt", "rb") as f:
        data = torch.load(f, weights_only=True)

    assert torch.equal(data["position"], sample_data["position"][None])
    assert torch.equal(data["time"], sample_data["time"][None])


def test_serial_dumper_metadata_is_saved(tmp_path_factory, sample_data):
    data_dir = tmp_path_factory.mktemp("data")
    dumper = SerialDumper(data_dir)
    dumper.start_episode()
    dumper.write(sample_data)
    dumper.end_episode(metadata={"robot_type": "franka"})

    with open(data_dir / "001.pt", "rb") as f:
        data = torch.load(f, weights_only=True)

    assert data["robot_type"] == "franka"


def test_serial_dumper_original_data_appending_same_tensor_raises_error(
    tmp_path_factory,
):
    data_dir = tmp_path_factory.mktemp("data")

    tensor = torch.tensor([1, 2, 3])

    dumper = SerialDumper(data_dir)
    dumper.start_episode()
    dumper.write({"tensor": tensor})

    with pytest.raises(AssertionError):
        dumper.write({"tensor": tensor})


def test_serial_dumper_metadata_keys_intersecting_with_data_keys_raises_error(
    tmp_path_factory,
):
    data_dir = tmp_path_factory.mktemp("data")
    dumper = SerialDumper(data_dir)
    dumper.start_episode()
    dumper.write({"tensor": torch.tensor([1, 2, 3])})
    with pytest.raises(AssertionError):
        dumper.end_episode(metadata={"tensor": 1})
