# How to install docker on Ubuntu computer
```bash
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Installing NVidia docker

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart docker
sudo systemctl restart docker

# Fix permissions
sudo usermod -aG docker $USER
newgrp docker   # Or logout/login

# Install UV locally to be able to regenerate dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Now you are ready to build our Docker.
```bash
docker/build.sh
```

## GR00T containers: uv mount caveat

If you customize `docker-compose.yml` volumes, **do not bind-mount** your host `~/.local/share/uv` into `/root/.local/share/uv` for `positro/gr00t` images.
GR00T's `/.venv/bin/python` can be a symlink into the image's own uv-managed CPython under `/root/.local/share/uv/python/...`, and the bind mount can hide that target and cause `/.venv/bin/python` to fail with `ENOENT`.
