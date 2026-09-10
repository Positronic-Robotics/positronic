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

## GR00T N1.7 containers

`make build-groot` builds the GR00T fork revision pinned in `Makefile`. The fork image contains
CUDA 12.8 and the upstream Python 3.12 environment at `/opt/gr00t-venv`.
Positronic installs its own locked environment at `/positronic/.venv`.
Both training and serving launch GR00T in its separate environment.

Build both images from Positronic:

```bash
make -C docker build-groot
IMAGE_TAG=local docker compose -f docker/docker-compose.yml run --rm --service-ports groot-server droid
```

Pass `GROOT_BASE_IMAGE=<image:tag>` to use an existing base and skip its build.
For local fork development, run `make -C docker build` in the fork,
then `make -C docker build-groot GROOT_BASE_IMAGE=positro/gr00t-base:local` in Positronic.

Do not mount host uv interpreter directories over the image's interpreter directories.
See [GR00T](../positronic/vendors/gr00t/README.md) for conversion, fine-tuning and inference.
