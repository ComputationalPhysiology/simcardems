# Graphical user interface

We also developed a lightweight graphical user interface for `simcardems` based on [streamlit](https://streamlit.io) and [`fenics_plotly`](https://pypi.org/project/fenics-plotly/).


## Running GUI in Docker
To run the GUI inside the docker container you need to forward the port where the GUI is running from the Docker container to your computer. The GUI is running by default on port 8501, and you can forward this port by running the container as follows

```
docker run --rm -p 8501:8501 -it ghcr.io/computationalphysiology/simcardems:
```
Please also consult [Run docker](docker.md) for more options on how to use Docker.

## Running the gui outside Docker

In order to run the gui you need to first install `streamlit` and `fenics_plotly`, which can be done by either installing these packages separately, i.e
```
python3 -m pip install streamlit fenics_plotly
```
or by installing `simcardems` with the extra requirements
```
python3 -m pip install "simcardems[gui]"
```

## Starting the gui

You can run the gui from the command line by executing the following command
```
python3 -m simcardems gui
```
This will open up a browser with an about page, and you can click on the `Simulation` radio button to get options about running a simulation. If no browser opens automatically, you can try to manually open the browser and go to the url http://localhost:8501


## Running a simulation

```{figure} figures/gui1.png
---
name: gui1
---
Options for running a simulation
```

If you select some options it will setup the models, and you can plot the mechanics mesh and the ep mesh inside the gui.

```{figure} figures/gui2.png
---
name: gui2
---
Visualize the mesh and run the simulation.
```

To run the simulation, you need to first create an output directory, which will be located in your home directory under a folder called `simcardems`, i.e `~/simcardems`. Each individual result will be stored in a separate subfolder which will be `results_<id>` where `id` is some hash of the input parameters. You can also change this to something different.


```{note}
These files can take up quite a lot of storage so it is a good idea to delete these folders from time to time
```

## Postprocessing the results
To look at the results you can click the `Postprocess` tab. Here you can select the folder you want to postprocess. By default it will list all the subfolder in the `simcardems` folder in your home directory.


```{figure} figures/gui3.png
---
name: gui3
---
Postprocess the results.
```
