# DreamRAM
## A Fine-Grained Configurable Design Space Modeling Tool for Custom 3D Die-Stacked DRAM

## Requirements
DreamRAM requires standard packages such as: numpy csv sys itertools json getopt os

For plotting paretos, DreamRAM uses matplotlib, mpl_toolkits, and paretoset:
```
pip install paretoset
```
## Usage
```
python3 dreamram.py [-m MEMORY_CONFIG] [-t TECH_CONFIG] [-o OUTPUT_LABEL]
```
The default output label is "default", i.e., the default output file is data/default/hbm3_default.csv. 
Other labels are given output directories and files in the same structure. 
Data is overwritten without checking existance.\nBe sure you have the unique output label if you do not want to overwrite.

For plotting:
```
python3 plot.py [-i INPUT_LABEL]
```
The input label should match the output label of your run. Plots are saved to the plot/ directory.