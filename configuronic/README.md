# Configuronic

Configuronic is a concise "Configuration as a Code" library that is developed by Positronic Robotics while working on Robotics ML platform.

We designed this library from the first principles, after fruitless attempts to cope with Hydra framework.

These foundatinoal principles are:
* Don't repeat yourself twice.
* Provide clearn and effective way to change configuration from the command line.
* Express configurations in Python, rather than YAML, JSON, XML, INI or anything else.
* Be minimal and simple.

Let's [take a look (FIXME)](https://positronic.ro/) into a

## Simple example
```python
# 'examples/basic.py'
import numpy as np
import configuronic as cfn


def noisy_sin(w, th, noise_std=0.1):
    t = np.linspace(0, 1, 100)
    return np.sin(w * t + th) + np.random.normal(0, noise_std, 100)


noisy_sin_01 = cfn.Config(noisy_sin, w=1, th=0)
clean_sin = noisy_sin.override(noise_std=0)


@cfn.config
def second_order_polynomial(a=1, b=0, c=0):
    x = np.linspace(0, 1, 100)
    return a * x**2 + b * x + c


def print_exp_moving_average(sequence, alpha=0.1):
    result = None
    for x in sequence:
        if result is None:
            result = x
        else:
            result = alpha * x + (1 - alpha) * result
        print(result)
    return result


main = cfn.Config(print_exp_moving_average, sequence=noisy_sin)

if __name__ == "__main__":
    cfn.cli(main)
```

And this is how you can run that script:
```bash
# Run default configuration
python utils/record_video.py --filename=/tmp/out.mp4
# This will print the following:
python utils/record_video.py --help
# TODO: Generate output

# This will print error because filename must be specfied
python utils/record_video.py

# Overwrite existing parameter
python utils/record_video.py --filename=/tmp/out.mp4 --codec='mpeg4'

# Overwrite nested parameter
python utils/record_video.py --filename=/tmp/out.mp4 --camera.fps=60

# TODO: What about None in cli arguments? How is it handled?

# Absolute reference to another object config
python utils/record_video.py --filename=/tmp/out.mp4 --camera=@utils.record_video.v4l_camera
# With it you can access it arguments
python utils/record_video.py --filename=/tmp/out.mp4 --camera=@utils.record_video.v4l_camera --camera.usb_path='/dev/v4l/another'
# This will fail, as the default camera does not have 'usb_path' argument
python utils/record_video.py --filename=/tmp/out.mp4 --camera.usb_path='/dev/v4l/another' --camera=@utils.record_video.v4l_camera

# Relative reference to another config
python utils/record_video.py --filename=/tmp/out.mp4 --camera=:basic_camera_1
# Relative reference to another config and modify it
python utils/record_video.py --filename=/tmp/out.mp4 --camera=:basic_camera_1 --camera.camera_id=2
```

## Configuration as a Code

When you "configure" something, what exactly do you configure? Function calls. Different configurations mean different set of arguments for a function call. Usually this is some kind of `main` function. It may call other functions, or instantiate classes, and you might want to configure them as well.

In our library, `class Config` is just a callable with the fixed set of arguments to be called. In functional paradigm this is often called closure [**give a link**].

Essentially you can do just two things with `Config` – `instantiate()` and `override()` (allright, you can also `copy()`, but it is not that cool as the other two).

`instantiate()` just calls the callable with bound arguments, with an important caveat. If any of the arguments is another `Config`, it instantiates it beforehand. Configuronic also recursively goes through lists, tuples and dictionary values, and if they are `Configs`, it instantiates them.

`override(**overrides)` creates another `Config` with bindings updated by `**overrides`. This is the part responsible for commandline magic, as we will see later. Now, let us consider a

## Best practices
* Do imports in configuration functions
* Specify defaults in the configuration, not in the function defaults
* If you want to put somewhere the set of arguments to remember the configuration, just store it as overwritten config in your main


## Design Principles

- **Config = Code**: No YAML, JSON, or other text-based configuration formats
- **Configuration is just the way to call functions**: Configurations are simply stored function calls that can be modified and executed later

## Key Features

- 🔧 **Deferred execution**: Store function calls and their arguments without executing them
- 🔄 **Override parameters**: Easily modify configurations without changing the original
- 🏗️ **Composable**: Nest configurations within other configurations
- 🖥️ **Auto CLI**: Automatically generate command-line interfaces from configurations
- 🔗 **Smart imports**: Support for both absolute (`@`) and relative (`:`) import paths
- 📝 **Type-safe**: Leverage Python's type system for configuration validation

## Installation

```bash
pip install configuronic
```

## Quick Start

### Basic Usage

```python
import configuronic as cfn

# Define a function
def train_model(learning_rate: float, batch_size: int, epochs: int):
    print(f"Training with lr={learning_rate}, batch={batch_size}, epochs={epochs}")
    return "model_trained"

# Create a configuration
model_config = cfn.Config(
    train_model,
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)

# Execute the configuration
result = model_config.instantiate()
```

### Using the Decorator

```python
import configuronic as cfn

@cfn.config(learning_rate=0.001, batch_size=32)
def train_model(learning_rate: float, batch_size: int, epochs: int = 100):
    print(f"Training with lr={learning_rate}, batch={batch_size}, epochs={epochs}")
    return "model_trained"

# Override parameters and execute
result = train_model.override(epochs=200).instantiate()
```

## Real-World Examples

### Hardware Configuration

```python
import configuronic as cfn

# Configure camera settings
@cfn.config(device_path="/dev/video0", width=1280, height=720, fps=30)
def camera_left(device_path: str, width: int, height: int, fps: int):
    from my_drivers.camera import Camera
    return Camera(device_path=device_path, width=width, height=height, fps=fps)

# Create a variant for the right camera
camera_right = camera_left.override(device_path="/dev/video1")

# Configure robot arm
@cfn.config(ip="192.168.1.100", speed_factor=0.5)
def robot_arm(ip: str, speed_factor: float):
    from my_drivers.robot import RobotArm
    return RobotArm(ip=ip, speed_factor=speed_factor)
```

### Nested Configurations

```python
import configuronic as cfn

# Main application configuration
def main(cameras: dict, robot, output_dir: str = "/tmp/data"):
    print(f"Starting with cameras: {cameras.keys()}")
    print(f"Robot: {robot}")
    print(f"Output: {output_dir}")

main_config = cfn.Config(
    main,
    cameras=cfn.Config(
        dict,
        left=camera_left,
        right=camera_right,
    ),
    robot=robot_arm,
    output_dir="/home/user/experiments"
)

# Override nested parameters
experiment_config = main_config.override(
    **{
        "cameras.left.fps": 60,          # Override nested camera FPS
        "robot.speed_factor": 0.8,       # Override robot speed
        "output_dir": "/tmp/experiment"   # Override output directory
    }
)

# Execute
experiment_config.instantiate()
```

### Import Paths

Configuronic supports two types of import paths:

#### Absolute Imports (`@`)
```python
# Import and instantiate an object from an absolute path
config = cfn.Config(
    my_function,
    processor="@sklearn.preprocessing.StandardScaler",  # Imports and creates StandardScaler()
    model="@torch.nn.Linear"
)
```

#### Relative Imports (`:`)
```python
# Import relative to the current module/class
@cfn.config(base_model="@transformers.BertModel")
def fine_tuned_model(base_model, num_classes: int):
    # Use `:` to import relative to BertModel
    tokenizer = ":BertTokenizer"  # Resolves to transformers.BertTokenizer
    return MyModel(base_model, tokenizer, num_classes)
```

### Command Line Interface

Automatically generate CLIs from your configurations:

```python
import configuronic as cfn

@cfn.config
def train(model_name: str, learning_rate: float = 0.001, epochs: int = 100):
    print(f"Training {model_name} for {epochs} epochs with lr={learning_rate}")

# This creates a CLI automatically
if __name__ == "__main__":
    cfn.cli(train)
```

Run from command line:
```bash
# Show help and current configuration
python train.py --help

# Override parameters
python train.py --model_name "bert-base" --learning_rate 0.0001 --epochs 50
```

### Environment-Specific Configurations

```python
import configuronic as cfn

# Base configuration
@cfn.config(host="localhost", port=8080, debug=False)
def server_config(host: str, port: int, debug: bool):
    return {"host": host, "port": port, "debug": debug}

# Development environment
dev_config = server_config.override(debug=True, port=8000)

# Production environment
prod_config = server_config.override(host="0.0.0.0", port=80)

# Staging environment
staging_config = prod_config.override(port=8080)
```

## Advanced Features

### Configuration Inspection

```python
# View configuration as YAML
print(config)

# Get required arguments that need to be set
required_args = cfn.get_required_args(config)
print(f"Missing arguments: {required_args}")

# Convert to dictionary
config_dict = config._to_dict()
```

### Error Handling

```python
from configuronic import ConfigError

try:
    # This will raise ConfigError if parameters are invalid
    result = config.instantiate()
except ConfigError as e:
    print(f"Configuration error: {e}")
```

### Override and Execute Pattern

```python
# Common pattern for creating flexible entry points
def run_experiment(**kwargs):
    return base_config.override(**kwargs).instantiate()

# Use with fire or other CLI libraries
import fire
fire.Fire(run_experiment)
```

## Best Practices

1. **Use type hints**: They help with IDE support and catch errors early
2. **Default values**: Provide sensible defaults in your functions
3. **Composition over inheritance**: Build complex configs by composing simpler ones
4. **Environment separation**: Create separate configs for dev/staging/prod
5. **CLI-ready**: Design configs to work well with the auto-generated CLI

## API Reference

### `Config`
- `Config(target, *args, **kwargs)`: Create a configuration
- `.override(**kwargs)`: Create a new config with overridden parameters
- `.instantiate()`: Execute the configuration and return the result
- `.copy()`: Create a deep copy of the configuration

### `@config` decorator
- `@config`: Convert a function into a Config object
- `@config(**kwargs)`: Create a Config with default parameters

### `cli(config)`
- Generate a command-line interface from a configuration

### Import Resolution
- `@module.Class`: Absolute import path
- `:RelativeClass`: Relative to the current context
- Multiple `:` traverse up the module hierarchy (`:../sibling_module.Class`)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]
