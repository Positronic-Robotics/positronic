import os
from hydra_zen import instantiate

import cfg.ui
import cfg.env
from tests.stub.ui import ui_stub
from tests.stub.hardware.camera import camera_stub
from data_collection_zen import main, dataset_dumper


def test_data_collection_produces_data_file(tmp_path):
    """Test main function with actual systems to ensure correct execution."""

    cfg.ui.ui_store.add_to_hydra_store()
    cfg.env.store.add_to_hydra_store()

    ui = instantiate(ui_stub)
    env = instantiate(cfg.env.umi, camera=camera_stub)
    data_dumper = instantiate(dataset_dumper(out_dir=tmp_path))

    main(ui=ui, env=env, data_dumper=data_dumper, rerun=False, sound=None)

    assert os.path.exists(tmp_path / "001.pt")
