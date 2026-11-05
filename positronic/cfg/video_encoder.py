import configuronic as cfn

from positronic.dataset.gst_video import GstH264Encoder, RawFormat
from positronic.dataset.video import LibavEncoder

# Jetson hardware H.264: nvvidconv converts to I420 on the VIC, and nvv4l2h264enc encodes on NVENC.
# 33_333 bits is 1 Mbit/s at 30 fps, which measured about 4.8 KiB per frame on SVGA ZED cameras.
jetson_h264 = cfn.Config(
    GstH264Encoder,
    encode=(
        *('nvvidconv', '!', 'video/x-raw(memory:NVMM),format=I420', '!'),
        *('nvv4l2h264enc', 'control-rate=1', 'bitrate={bitrate_bps}', 'iframeinterval={gop}', 'idrinterval={gop}'),
        *('insert-sps-pps=true', 'num-B-Frames=0'),
    ),
    elements=('nvvidconv', 'nvv4l2h264enc'),
    frame_bits=33_333,
    raw_format=RawFormat.BGRX,
)

# libx264 on one core, for a host without the hardware encoder
libx264_veryfast = cfn.Config(LibavEncoder, options=(('preset', 'veryfast'), ('threads', '1')))

# NVIDIA desktop GPU H.264 through h264_nvenc, sized like libx264 at its defaults on 720p camera streams
nvenc_h264 = cfn.Config(
    LibavEncoder,
    codec='h264_nvenc',
    options=(
        ('preset', 'p4'),
        ('profile', 'high'),
        ('rc', 'vbr'),
        ('cq', '35.75'),
        ('b', '0'),
        ('bf', '3'),
        ('b_ref_mode', 'middle'),
        ('spatial_aq', '1'),
        ('temporal_aq', '1'),
        ('rc-lookahead', '20'),
    ),
)
