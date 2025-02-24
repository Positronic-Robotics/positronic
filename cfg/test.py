from hydra_zen import builds, zen, ZenStore

import cfg.hardware.camera

import ironic as ir

async def _main(camera: ir.ControlSystem):
    from tools.video import VideoDumper

    # camera = LinuxPyCamera('/dev/video0')
    system = ir.compose(
        camera,
        VideoDumper("video.mp4", 30, codec='libx264').bind(
            image=ir.utils.map_port(lambda x: x['video2.image'], camera.outs.frame)
        )
    )
    await ir.utils.run_gracefully(system)


def main(camera: ir.ControlSystem):
    import asyncio
    asyncio.run(_main(camera))


store = ZenStore()
store(
    builds(
        main,
        populate_full_signature=True,
        hydra_defaults=["_self_", {"hardware/cameras@camera": "umi_merged"}]
    ),
    name="camera_test"
)

if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(main).hydra_main(config_name="camera_test", version_base="1.1", config_path=".")
