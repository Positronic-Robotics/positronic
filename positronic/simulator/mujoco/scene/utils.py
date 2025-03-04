import glob
import multiprocessing
import os


def gen_fn(model_path):
    # TODO: robosuite's OpenGL is conflicting with other libraries OpenGL (e.g. dearpygui)
    # so we need to run this in a separate process
    from simulator.mujoco.scene.scene_builder import construct_scene
    construct_scene(model_path)


def generate_scene_in_separate_process(output_dir: str) -> str:
    assert output_dir is not None, "data_output_dir must be specified to build the scene"

    scene_dir = os.path.join(output_dir, "scenes")
    os.makedirs(scene_dir, exist_ok=True)

    next_scene_idx = len(glob.glob(f"{scene_dir}/*.xml"))
    new_model_path = os.path.join(scene_dir, f"scene_{next_scene_idx:04d}.xml")

    print(f"Building scene to {new_model_path}")

    process = multiprocessing.Process(target=gen_fn, args=(new_model_path,))
    process.start()
    process.join()

    return new_model_path
