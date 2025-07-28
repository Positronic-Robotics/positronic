# Configuronic

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/configuronic.svg)](https://badge.fury.io/py/configuronic)
**FIXME: Add tests**

**Configuronic** is a simple yet powerful "Configuration as Code" library designed for modern Python applications, particularly in robotics, machine learning, and complex system configurations. Born from the need for a cleaner alternative to existing configuration frameworks, configuronic embraces Python's native syntax while providing powerful CLI integration and hierarchical configuration management.

## ✨ Why Configuronic?

* 🎯 **DRY Principle**: Write configurations in Python, not YAML/JSON/XML
* 🚀 **CLI-First**: Automatic command-line interfaces with complex nested parameter support
* 🔧 **Simple & Minimal**: Clean API that gets out of your way
* 🌳 **Hierarchical**: Deep nesting and inheritance support
* 🔄 **Dynamic**: Runtime configuration resolution with relative imports

## 🚀 Quick Start

```python
import configuronic as cfn

@cfn.config(learning_rate=0.001, epochs=100)
def train_model(learning_rate: float, epochs: int, model_name: str = "bert-base"):
    print(f"Training {model_name} for {epochs} epochs with lr={learning_rate}")
    # Your training logic here

if __name__ == "__main__":
    cfn.cli(train_model)
```

Run from command line:
```bash
# Use defaults
python train.py

# Override parameters
python train.py --learning_rate=0.0001 --epochs=50 --model_name="gpt-2"

# See current configuration
python train.py --help
```

## 📖 Table of Contents

- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Real-World Examples](#real-world-examples)
- [Advanced Features](#advanced-features)
- [CLI Usage](#cli-usage)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Contributing](#contributing)

## 📦 Installation

Using pip
```bash
pip install configuronic
```

### Development Installation
```bash
git clone https://github.com/positronic/configuronic.git
cd configuronic
uv pip install -e .
```

## 🧠 Core Concepts

### Configuration as Code

In configuronic, configurations are **closures** - callables that store both the function and its arguments. It is somewhat similar to [`functools.partial`](https://docs.python.org/3/library/functools.html#functools.partial) This functional approach enables powerful composition and inheritance patterns.

```python
import configuronic as cfn

# Create a configuration
@cfn.config(batch_size=32, lr=0.001)
def create_optimizer(batch_size: int, lr: float):
    return torch.optim.Adam(lr=lr)

# Override and create variants
fast_optimizer = create_optimizer.override(lr=0.01)
large_batch_optimizer = create_optimizer.override(batch_size=128)

# Instantiate when needed
optimizer = fast_optimizer.instantiate()
```

### Two Main Operations

**1. `override(**kwargs)`** - Create configuration variants
```python
base_config = cfn.Config(MyModel, layers=3, units=64)
deep_config = base_config.override(layers=6)
wide_config = base_config.override(units=128)
```

**2. `instantiate()`** - Execute function with configured arguments and get its result
```python
model = deep_config.instantiate()  # Returns MyModel(layers=6, units=64)
```

**Callable Syntax** - Config objects are callable, providing a shorthand for override + instantiate
```python
# These are equivalent:
result1 = config.override(param=value).instantiate()
result2 = config(param=value)()
```

### Nested Configuration Override

Support for deep parameter modification using dot notation:

```python
# Configure a complex training pipeline
training_cfg = cfn.Config(
    train_pipeline,
    model=cfn.Config(TransformerModel, layers=6, hidden_size=512),
    optimizer=cfn.Config(torch.optim.Adam, lr=0.001),
    data=cfn.Config(DataLoader, batch_size=32)
)

# Override nested parameters
fast_training = training_cfg.override(**{
    "optimizer.lr": 0.01,
    "data.batch_size": 64,
    "model.layers": 12
})
```

## 🌍 Real-World Examples

### Robotics Hardware Configuration

```python
import configuronic as cfn

@cfn.config(ip="172.168.0.2", port="/dev/ttyUSB0")
def robot_arm(ip: str, relative_dynamics_factor: float = 0.2):
    from my_robots import FrankaArm
    return FrankaArm(ip=ip, dynamics_factor=relative_dynamics_factor)

@cfn.config(device_path="/dev/video0", fps=30)
def camera(device_path: str, width: int = 1920, height: int = 1080, fps: int = 30):
    from my_cameras import Camera
    return Camera(device_path, width, height, fps)

# Create specific hardware configurations
left_camera = camera.override(device_path="/dev/video1")
right_camera = camera.override(device_path="/dev/video2")

# Main system configuration
@cfn.config(arm=robot_arm,
            cameras={'left': left_cam, 'right': right_cam})
def main(arm, cameras, gripper=None):
    from robot_library import RobotSystem
    system = RobotSystem(arm=arm, cameras=cameras, gripper=gripper)
    system.run()

if __name__ == "__main__":
    cfn.cli(main)
```

### Machine Learning Pipeline

```python
@cfn.config(model_name="bert-base", max_length=512)
def create_tokenizer(model_name: str, max_length: int):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = max_length
    return tokenizer

@cfn.config(hidden_size=768, num_layers=12)
def create_model(hidden_size: int, num_layers: int, tokenizer):
    vocab_size = len(tokenizer)
    return TransformerModel(vocab_size, hidden_size, num_layers)

# Configure the complete pipeline
@cfn.config()
def training_pipeline(
    tokenizer=create_tokenizer,
    model=create_model,
    learning_rate: float = 1e-4,
    batch_size: int = 16
):
    # Pipeline implementation
    return TrainingPipeline(tokenizer, model, learning_rate, batch_size)

if __name__ == "__main__":
    cfn.cli(training_pipeline)
```

Run with different configurations:
```bash
# Use defaults
python train.py

# Quick experiments
python train.py --learning_rate=1e-3 --batch_size=32

# Override nested model parameters
python train.py --model.num_layers=6 --tokenizer.max_length=256

# Switch to different model entirely
python train.py --tokenizer.model_name="gpt2" --model.hidden_size=1024
```

## 🔧 Advanced Features

### Import Resolution with `@` and `.`

Configuronic provides powerful import resolution syntax that allows you to dynamically reference Python objects, especially useful for CLI usage.

#### Absolute Imports (`@`)
Direct import paths to any Python object. If you need to use a literal `@` at the beginning of a string (not for imports), use `@@`:
```bash
# From command line - these import exact module paths
python train.py --model="@transformers.BertModel"     # Import BertModel
python train.py --message="@@starts_with_at"          # Literal string "@starts_with_at"
python train.py --text="foo@bar"                      # No escaping needed in the middle
```

#### Relative Imports (`.`)
Navigate relative to the current module, similar to Python's relative import syntax:

```python
# If default is myproject.models.BertEncoder
python train.py --encoder=".RobertaEncoder"        # -> myproject.models.RobertaEncoder (same module)
python train.py --encoder="..utils.CustomEncoder"  # -> myproject.utils.CustomEncoder (parent module)
python train.py --encoder="...shared.BaseEncoder"  # -> myproject.shared.BaseEncoder (grandparent module)
```

**How it works:** Each `.` acts like `../` in file system navigation:
- `.` = stay in current module (like `./`)
- `..` = go up one module level (like `../`)
- `...` = go up two module levels (like `../../`), etc.

The path after the dots specifies the target within that module hierarchy.

#### Configuration Copy Across Modules

The `copy()` method updates module context so relative imports (`.`) resolve from the new module location:

```python
# configs/base.py
original_config = cfn.Config(SomeClass, value=1)

# experiments/vision.py
from configs.base import original_config

# Copy updates the config's module context to experiments.vision
copied_config = original_config.copy()

@cfn.config()
def local_function():
    return "local result"

# When copied_config is used as default, '.' resolves in experiments.vision
env_cfg = cfn.Config(Environment, setup=copied_config)
specialized_cfg = env_cfg.override(setup=".local_function")  # Finds local_function
```

Without `copy()`, `.local_function` would try to resolve in `configs.base` and fail.

### Lists and Dictionaries

Configuronic seamlessly handles nested data structures:

```python
simulation_cfg = cfn.Config(
    run_simulation,
    loaders=[
        cfn.Config(AddCameras, camera_config=camera_cfg),
        cfn.Config(AddObjects, objects=["cube", "sphere"]),
        cfn.Config(SetLighting, intensity=0.8)
    ],
    cameras={
        'main': cfn.Config(Camera, position=[0, 0, 1]),
        'side': cfn.Config(Camera, position=[1, 0, 0])
    }
)

# Override specific items
modified_sim = simulation_cfg.override(**{
    "loaders.0.camera_config.fps": 60,  # First loader's camera FPS
    "cameras.main.position": [0, 0, 2]   # Main camera position
})
```

### Configuration Inheritance

```python
# Base configuration
base_camera = cfn.Config(Camera, width=1920, height=1080, fps=30)

# Derived configurations
hd_camera = base_camera.override(width=1280, height=720)
high_fps_camera = base_camera.override(fps=60)
webcam = base_camera.override(width=640, height=480, fps=15)

# All inherit base settings unless overridden
```

## 🖥️ CLI Usage

Configuronic leverages [Python Fire](https://github.com/google/python-fire) for automatic CLI generation:

### Basic CLI
```python
@cfn.config(param1="default", param2=42)
def my_function(param1: str, param2: int):
    return f"{param1}: {param2}"

if __name__ == "__main__":
    cfn.cli(my_function)
```

### Command Line Examples
```bash
# Show help and current config
python script.py --help

# Override parameters
python script.py --param1="hello" --param2=100

# Nested parameter override
python script.py --model.layers=6 --optimizer.lr=0.001

# Using absolute imports
python script.py --model="@my_models.CustomTransformer"

# To pass a string that starts with @, repeat it twice
python script.py --message="@@this_is_literal_at_sign"

# Using relative imports
python script.py --tokenizer=".CustomTokenizer"

# Complex nested overrides
python script.py --cameras.left.fps=60 --cameras.right.device="/dev/video2"
```

### Parameter Override Order ⚠️

**Important:** Parameter overrides are executed in order of declaration. When overriding nested configurations, set the parent object first, then its properties:

```bash
# ✅ Correct: set camera first, then its resolution
python script.py --camera="@opencv.Camera" --camera.resolution="full_hd"

# ❌ Incorrect: this will reset camera after setting resolution
python script.py --camera.resolution="full_hd" --camera="@opencv.Camera"
```

In the incorrect example, the default camera's resolution gets updated first, but then the entire camera object is replaced, losing the resolution override.

## 📚 API Reference

### Core Classes

#### `Config(target, *args, **kwargs)`
Main configuration class that stores a callable and its arguments.

**Methods:**
- `override(**kwargs) -> Config`: Create new config with updated parameters
- `instantiate() -> Any`: Execute the configuration and return result
- `copy() -> Config`: Deep copy the configuration
- `__call__(**kwargs) -> Any`: `override` config with `**kwargs` and `instantiate` it. **Note:** only keyword specified arguments are supported.

#### `@config` Decorator
```python
@cfn.config  # No override, just turn function into config.
def print_greeting(greeting: str = 'Hello', entity: str = 'world'):
    print(f'{greeting} {entity}!')

@cfn.config(arg1="default", arg2=42)  # With defaults
def my_function(...):
    pass
```

### Utility Functions

#### `cli(config: Config)`
Generate automatic command-line interface for any configuration.

#### `get_required_args(config: Config) -> List[str]`
Get list of required arguments for a configuration.

### Special Syntax

- `@module.path.Class` - Absolute import path to any Python object
- `.RelativeClass` - Relative import (same module, like `./`)
- `..parent.Class` - Relative import (up one level, like `../`)
- `@@literal_string` - Escape literal `@` characters in the beginning of strings

> **Path Resolution:** The `.` syntax works like file system navigation where each dot moves up one module level in the Python package hierarchy, then navigates down to the specified target.


## 💡 Best Practices

### 1. Create Separate Modules for Configurations
If you don't want your business logic modules to depend on `configuronic`, it's wise to have a separate package for configurations.
```python
# configs/models.py
transformer_base = cfn.Config(TransformerModel, layers=6, hidden_size=512)
transformer_large = transformer_base.override(layers=12, hidden_size=1024)

# configs/training.py
from .models import transformer_base
training_pipeline = cfn.Config(TrainingPipeline, model=transformer_base)
```

### 2. Import Inside Configuration Functions
In robotic applications, some configurations may depend on parituclar hardware and Python packages that provide drivers, that are not always available. If you don't want to force your users to install all of them, consider making imports inside the functions that you configure.
```python
@cfn.config()
def create_model(layers: int = 6):
    from my_project.models import TransformerModel
    return TransformerModel(layers=layers)
```

But in smaller projects it might be very convinient to put configurations alongside the methods they manage.

### 3. Use `override` to create as many custom configurations as you need
When working with text configuration files, it's natural to create separate files for different environments or use cases. In `configuronic`, just create new configuration variables that override the base config.

```python
# Base training configuration
base_training = cfn.Config(
    TrainingPipeline,
    model=cfn.Config(TransformerModel, layers=6, hidden_size=512),
    optimizer=cfn.Config(torch.optim.Adam, lr=0.001),
    batch_size=32,
    epochs=10
)

# Development environment - smaller, faster
dev_training = base_training.override(
    batch_size=8, epochs=2,
    **{"model.layers": 3, "model.hidden_size": 256})

# Production environment - optimized settings
prod_training = base_training.override(
    batch_size=64, epochs=100,
    **{"optimizer.lr": 0.0001})

# Experimental setup - large model
experimental_training = base_training.override(
    **{
        "model.layers": 12,
        "model.hidden_size": 1024,
        "optimizer.lr": 0.0005,
        "batch_size": 16
    })

# Quick debugging setup
debug_training = base_training.override(
    epochs=1, batch_size=2, **{"model.layers": 1})

# Now you can easily switch between configurations
# TODO: Make this part of Configuronic
if __name__ == "__main__":
    import sys
    configs = {
        'dev': dev_training,
        'prod': prod_training,
        'experimental': experimental_training,
        'debug': debug_training
    }

    config_name = sys.argv[1] if len(sys.argv) > 1 else 'dev'
    selected_config = configs.get(config_name, dev_training)

    cfn.cli(selected_config)
```

**Usage:**
```bash
python train.py dev          # Use development config
python train.py prod         # Use production config
python train.py experimental # Use experimental config
python train.py debug        # Use debug config

# Still supports all override capabilities
python train.py prod --epochs=50 --batch_size=128
```


## 🤝 Contributing
We welcome contributions! Here's how to get started:

### Development Setup
```bash
git clone https://github.com/your-org/configuronic.git
cd configuronic
uv pip install -e ".[dev]"  # FIXME: add pytest and pytest-cov there
```

### Running Tests
```bash
pytest  # Run all tests
pytest --cov=configuronic  # Run with coverage
```
### 📋 Guidelines
- Follow existing code style and patterns
- Add tests for new functionality
- Ensure all tests pass before submitting
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/your-org/configuronic/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-org/configuronic/discussions) **FIXME: Create Discord**
- 📧 **Email**: hi@positronic.ro

---

**⭐ If you find Configuronic useful, please consider giving it a star on GitHub!**
