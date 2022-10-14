(section:docker-install)=
# Install with Docker

The preferred way to install `simcardems` is through [Docker](https://docs.docker.com/get-docker/).

This is a good choice if you want to use `simcardems` in an isolated environment.

We provide both a pre-built docker image which you can get by pulling from [GitHub Container Registry](https://github.com/computationalphysiology/simcardems/pkgs/container/simcardems)
```
docker pull ghcr.io/computationalphysiology/simcardems:latest
```
These images are built to support both AMD64 and ARM64 architectures.

You can read more about how to run `simcardems` within a Docker container in [Run using Docker](docker.md)

## Building your own docker image

An alternative to pulling the image is to build it yourself.
We provide a Dockerfile in the root of the repo that contains all the instructions for building the docker image.
You can do this by executing the following command in the root folder of the project
```
docker build -t simcardems .
```
This will create a docker image with the name `simcardems`.
