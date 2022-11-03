# Data visualization

## Plot traces of single node
For quick visualization of results, traces extracted from the center of the tissue can be plotted with the command
```
$ python3 -m simcardems postprocess "path_to_results" --plot-state-traces
```
The `path_to_results` directs to the folder containing the `results.h5` file.
A figure is generated showing traces of membrane potential ($V$), intracellular calcium concentration ($Ca$), stress ($\lambda$) and active tension ($T_a$). This figure is saved in the folder containing the original data.

## Create XDMF-files
The results of a simulation with `simcardems` are stored in a single file called `results.h5`. These results can be visualized in ParaView, but require 1 step in between: Creation of XDMF-files from `results.h5`. Execute
```
$ python3 -m simcardems postprocess "path_to_results" --make-xdmf
```
to create XMDF-files of all stored variables.
Each XDMF-file is named after the variable, and has a h5-file with the corresponding name.

## Visualization in ParaView
The XDMF-files can be opened in ParaView using XDMF3ReaderS or XDMF3ReaderT, but require the corresponding h5-file to be present in the same folder.
