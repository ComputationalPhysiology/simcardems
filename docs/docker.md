# Run using docker

If you want to use docker please make sure to follow the installations step for how to install with {ref}`section:docker-install`.

## Creating the container
Once you have the `simcardems` docker image you can create the container as follows

```
docker run --name simcardems -v "$(pwd)":/app -p 8501:8501 -dit ghcr.io/computationalphysiology/simcardems
```
This will create a a new container (aka a virtual machine) that you can use to execute the scripts.
Note that after executing the `docker run` command, the container will be created and it will run in the background (daemon-mode).

## Execute command line scripts

You can now execute the command line script using the command
```
docker exec -it simcardems python3 -m simcardems
```
For example
```
 docker exec -it simcardems python3 -m simcardems run --help
```
or
```
 docker exec -it simcardems python3 -m simcardems run -T 1000
```

## Stopping the container

When you are done using the script, you should stop the container so that it doesn't take up resources on your computer. You can do this using the command
```
docker stop simcardems
```

## Starting the container again

To start the container again you can execute the command
```
docker start simcardems
```
You can now do ahead the [execute the command line scripts](#execute-command-line-scripts) again.

## Deleting the container

If you don't want to use the container anymore, of your need to rebuild the image because there has been updates to the `simcardems` package, you can delete the container using the command
```
docker rm simcardems
```
Note that in order to use the container again, you need to first [create the container](#creating-the-container).
