# Build and test on Serverless AI

A Nebius Serverless job can test a policy image, and it can build one.
The competition steps stay in [Enter the Physical AI Competition](nebius-competition.md).
Register, enter and submit on the computer that holds the API key.

## What you need

- Install the [Nebius CLI](https://docs.nebius.com/cli/install), and run `nebius profile create`.
  The job uses the project of the profile.
- Make an SSH key with `ssh-keygen -t ed25519` when you have none.
- Keep one public IP address free in the project. An SSH key on a job allocates one.

## Test the image

[`docker/test_policy_image_serverless.sh`](../docker/test_policy_image_serverless.sh) runs your image as a
job. It calls `POST /api/v1/keepalive` on port 8000 from the machine under the job. Then it deletes the job.

```bash
bash docker/test_policy_image_serverless.sh docker.io/<you>/<image>:v1
bash docker/test_policy_image_serverless.sh --gpu docker.io/<you>/<image>:v1
```

The script sets `AUTH_TOKEN` to `test`. It does not replace the image command.
The image `ENTRYPOINT` starts the server.

The job has a network for the whole run.
Run [`docker/test_policy_image.sh`](../docker/test_policy_image.sh) to test the image with no egress.
That script needs Docker on amd64 Linux. The [build job](#build-on-a-serverless-job) is one place to run it.

The script prints one of these lines, and the end of the log:

- `the server answered: {"alive_seconds": ...}`: the server loaded its weights. The test passed.
- `the server exited`: read the log. An image that needs a GPU stops on a CPU job.
- `the server answered HTTP <code>`: the server refused the request. A `401` means the token is wrong.
- `the server answered 200 without {"alive_seconds": ...}`: the keepalive body must contain `alive_seconds`.
- `the server listens only on 127.0.0.1`: make the server listen on `0.0.0.0`.
- `no answer in 900 s`: the server did not answer. With `--gpu`, the wait is 600 s.

A private image needs a registry user and a password file. Run these commands in your own terminal:

```bash
bash docker/test_policy_image_serverless.sh --registry-username=<user> \
    --registry-password-file=<file> docker.io/<you>/<image>:v1
```

The script reads the password from the file. It removes one line ending from the end of the file.
The Nebius CLI takes the password as an argument of `nebius ai job create`.

With no `--gpu`, the job is `cpu-d3` and `8vcpu-32gb`.
With `--gpu`, the job is `gpu-l40s-a` and `1gpu-8vcpu-32gb`.
The disk is 500 GiB.
`NEBIUS_JOB_NAME`, `NEBIUS_PLATFORM`, `NEBIUS_PRESET`, `NEBIUS_DISK_SIZE` and `NEBIUS_SSH_KEY` override these.
The header of the script lists each default.

The job timeout is 1 hour. The script deletes the job when the test ends.
A large image must pull and answer inside that hour.

`docker.io/positro/pi05-droid` is the first image to run. Use `--gpu` for it.
It uses about 10 GB of GPU memory and 20 GB of memory.

## Build on a Serverless job

A Jupyter Devlab cannot build an image. The Devlab runs one container, and that container has no Docker.
A job container has no Docker either. `nebius ai job ssh` opens the job container.

The machine under the job has Docker. Build the image there.
[`docker/nebius_serverless.sh`](../docker/nebius_serverless.sh) starts a job that stays up, and deletes it.

```bash
git clone https://github.com/Positronic-Robotics/positronic
bash positronic/docker/nebius_serverless.sh create
```

The command prints `ssh nebius@<address>`. Connect with that command.
The script adds `nebius` to the `docker` group before it prints the command.
`docker` works in that connection. `docker/test_policy_image.sh` calls `docker`, so this group is required.

On the host, clone the repository and build the image.
[Build and push your image](nebius-competition.md#build-and-push-your-image) gives the build command.
Then run the offline test on the host:

```bash
bash docker/test_policy_image.sh docker.io/<you>/<image>:v1
bash docker/test_policy_image.sh --gpu docker.io/<you>/<image>:v1
```

Push the image before you delete the job. The delete removes the disk. The local image goes with the disk.

```bash
bash positronic/docker/nebius_serverless.sh delete
```

The job has no stop. Nebius bills the job until you delete it, also when it does nothing.
The default timeout is 8 hours. Set `NEBIUS_TIMEOUT` to change it. The shortest timeout is 1 hour.

A stopped Devlab keeps the files under its workspace path.
An image you build with Docker on the host is not in that workspace.
A stop releases the machine, and that image is gone.

The default job is `gpu-l40s-a` and `1gpu-8vcpu-32gb`, with a 500 GiB disk.
For a job with no GPU, set `NEBIUS_PLATFORM=cpu-d3` and `NEBIUS_PRESET=8vcpu-32gb`.
The header of the script lists the other settings.

If Nebius has no capacity, the command deletes the job. Run it again later.
If Nebius refuses the job because of a quota, ask Nebius for a larger quota.

## What this path does not match

The platform runs the image with no network. A Serverless job has a network.
The offline test on the build host is the test that denies egress.

The platform GPU is one 40 GB slice of an H100.
The default GPU job here is one L40S with 48 GB of GPU memory.
Keep the model under 40 GB. Read `nvidia-smi` on the host while the model is loaded.

The platform gives 14 CPU cores and 150 GiB of memory.
The default GPU preset has 8 cores and 32 GiB of memory.
