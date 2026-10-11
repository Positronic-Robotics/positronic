#!/usr/bin/env bash
set -euo pipefail

# Preserve every package version chosen by the native model environment.
/opt/gr00t-venv/bin/python -c \
  'from importlib.metadata import distributions; print("\n".join("{}=={}".format(d.metadata["Name"], d.version) for d in distributions()))' \
  > /tmp/gr00t-constraints.txt
UV_OFFLINE=0 uv pip install --python /opt/gr00t-venv/bin/python --constraint /tmp/gr00t-constraints.txt \
  /opt/positronic-packages/wire '/opt/positronic-packages/model_server[websocket,grpc]' \
  configuronic==0.7.0 pos3==0.3.1
uv pip check --python /opt/gr00t-venv/bin/python
rm /tmp/gr00t-constraints.txt

# Source directories are populated after installation to keep dependency layers cached.
server_site=$(/opt/gr00t-venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' /opt/positronic-packages/wire /opt/positronic-packages/model_server \
  > "$server_site/positronic-server-source.pth"
