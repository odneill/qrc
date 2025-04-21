# Quantum Reservoir Computer (QRC) Simulations

This repository contains the code replicating the results in the paper "Photon Number-Resolving Quantum Reservoir Computing" [[10.1364/OPTICAQ.553294](https://doi.org/10.1364/OPTICAQ.553294)].

## Reproduce the paper's results

On Linux, with Conda installed:
```bash
bash run_all.sh
```
Paper figures will be in [figures/](figures/) and raw data in [experiments/](experiments/).

We recommend a sizable CPU and RAM. The results in the paper were generated using a machine with 28 cores and 256GB of RAM, and each `definition.py` file generates on order 100GB intermediate files. Less capable machines may be used by reducing the number of `WORKERS` and increasing the `split_factor`s in each `definition.py` file (see [experiments/](experiments/)).

Running 
```bash
bash run_all.sh clean
```
will run all experiments, but delete some larger intermediate files to save disk space.


### Setup

If you are unable to run the `run_all.sh` script, you can set up the environment manually.

We recommend using the provided environment file to create a new Conda environment. [Miniforge](https://github.com/conda-forge/miniforge) is recommended for this purpose. 

From root project directory:
```bash
conda env create -n qrc_paper -f environment.yml
conda activate qrc_paper
pip install .
```

#### Perceval

While we do rely on the [Perceval](https://github.com/Quandela/Perceval) library for the implementation of the SLOS algorithm, we have made some modifications which are requirements for running this code.
 - Support for pickling of Exqalibur state objects.
 - Support for vectorial probabilities/ amplitudes in these states, for efficient parallel tensor product operations.
  
These are performed in [src/qrc/_compat.py](src/qrc/_compat.py) where we monkey patch the specific version of Perceval given in the [pyproject.toml](pyproject.toml) file. Other versions of Perceval will likely work with some modification to ensure the patches are applied correctly.


### Experiments

Each directory in [experiments](experiments) contains a `definition.py` file, which defines the particular experiment(s) to perform. 
To run a particular experiment, navigate to the directory containing the `definition.py` file and run the necessary tools from `qrc.tools`, i.e.
```bash
python -m qrc.tools.run_experiments
python -m qrc.tools.postprocess
python -m qrc.tools.eval_tasks
```

### Figures

Once the data has been generated in the experiments directories, the figures for the paper are generated using the scripts in [figures](figures).

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{Nerenberg:25,
  author = {Sam Nerenberg and Oliver D. Neill and Giulia Marcucci and Daniele Faccio},
  journal = {Optica Quantum},
  number = {2},
  pages = {201--210},
  publisher = {Optica Publishing Group},
  title = {Photon number-resolving quantum reservoir computing},
  volume = {3},
  month = {Apr},
  year = {2025},
  url = {https://opg.optica.org/opticaq/abstract.cfm?URI=opticaq-3-2-201},
  doi = {10.1364/OPTICAQ.553294},
}
```
