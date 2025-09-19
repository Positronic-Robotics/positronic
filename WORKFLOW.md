# Example workflow
This document guides you through the workflow of training a robot to perform a task via immitationg learning, using Positronic project.

## Installation
First, you need to set up the environment
```bash
git clone git@github.com:Positronic-Robotics/positronic.git
cd positronic

uv venv -p 3.11
source .venv/bin/activate
uv sync --frozen --extra dev
```

If you are going to run the policy on the real hardware (we support Franka and Kinova arms, as well as SO101 in the follower mode),
you need to install hardware deps
```bash
uv sync --frozen --extra hardware
```

## Collecting the data

We have a data_collection.py dedicated to this. This is a script that lets a user to control robot (either real or simulated one) to perform operations
and write down the data in Positronic Dataset format.

```bash
python -m positronic.data_collection sim --sound=None --webxr=.iphone --output_dir=~/datasets/[YOUR_DATASET]
```

By default, this start simulator, where you control the virtual Franka arm. You will see the table with green and red cubes. Your goal is to collect demonstrations of picking the green cube and put it on top of the red ones.

TODO: Describe how you can control using WebXR on your phone.
TODO: Describe that you also may use a VR Helmet (we tested on oculus) and how you control the arm. Describe what needs to be changed in the command line (`--webxr=.oculus`).


## Visualising the data
For your convenience, we created a Python web server that lets you to navigate through dataset and see its content. First, let us install the extras we need for it
```bash
uv sync --frozen --extra server
```
And now we can run the server
```bash
python -m positronic.server.positronic_server --root=~/datasets/[YOUR_DATASET] --port=5001
```
Go to https://localhost:5001/ (or straight to https://localhost:5001/episode/0) and see it. We use a great [rerurn](https://rerun.io) library to power visualisations.

## Coverting data to LeRobot format
Currently we employ [LeRobot](https://github.com/huggingface/lerobot) library for traning scripts and our [Positronic dataset format](positronic/dataset/README.md) is yet to support LeRobot training scripts directly. Hencewe have to convert to LeRobot native dataset. This is temporary solution, as we will support training directly on Positronic dataset.

In order to do that conversion, use [`to_lerobot.py`](positronic/training/to_lerobot.py) script with the following commandline:
```bash
uv run --with-editable . -s positronic/training/to_lerobot.py convert \
    --input_dir=~/datasets/[YOUR_DATASET] \
    --output_dir=~/datasets/lerobot/[YOUR_DATASET] \
    --task="pick up the green cube and place it on the red cube"
```
