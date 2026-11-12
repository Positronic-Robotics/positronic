import pytest

pytest.importorskip('transformers')

from positronic.vendors.molmoact2 import server  # noqa: E402


@pytest.mark.parametrize('pipeline', [server.droid, server.droid_3cam, server.yam_bimanual])
def test_every_pipeline_asks_the_client_for_jpeg_frames(pipeline):
    assert pipeline.instantiate().compress_images is True
