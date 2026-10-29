from functools import partial

import configuronic as cfn

from positronic.cfg import video_encoder
from positronic.dataset.local_dataset import LocalDatasetWriter

local = cfn.Config(partial, LocalDatasetWriter)
jetson = local.override(video_encoder=video_encoder.jetson_h264)
