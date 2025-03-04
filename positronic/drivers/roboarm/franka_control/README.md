# Install

In order to install that, just run.

You have to have `Eigen`, `libfranka` and `pybind11` installed.

```bash
conda activate positronic  # Most likely this is already done
pushd hardware/franka_control
pip install -e .
popd
```
