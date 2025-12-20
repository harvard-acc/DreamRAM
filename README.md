# DreamRAM
## Introduction
DreamRAM is a Fine-Grained Configurable Design Space Modeling Tool for Custom 3D Die-Stacked DRAM. DreamRAM analytically models bandwidth, capacity, energy, latency, and area while exposing fine-grained design parameters at the MAT, subarray, bank, and inter-bank levels. By providing a unified and extensible exploration framework, DreamRAM enables researchers and designers to uncover new opportunities for workload-tailored memory design.

## Usage
### Requirements
DreamRAM requires standard packages such as: numpy csv sys itertools json getopt os

For plotting paretos, DreamRAM uses matplotlib, mpl_toolkits, and paretoset:
```
pip install paretoset
```
### Running DreamRAM
```
python3 dreamram.py [-m MEMORY_CONFIG] [-t TECH_CONFIG] [-o OUTPUT_LABEL]
```
The default output label is "default", i.e., the default output file is data/default/hbm3_default.csv. Other labels are given output directories and files in the same structure. Data is overwritten without checking existance. Be sure you have the unique output label if you do not want to overwrite your previous run.

### Generating Plots
```
python3 plot.py [-i INPUT_LABEL]
```
The input label should match the output label of your run. Plots are saved to the plot/ directory.

## Citation
DreamRAM has been accepted to DATE 2026: [https://arxiv.org/abs/2512.12106](https://arxiv.org/abs/2512.12106)

## Authors
Victor Cai, Jennifer Zhou, Haebin Do, David Brooks, and Gu-Yeon Wei

Harvard University, 2025

Please direct further inquiries to victorcai@college.harvard.edu
