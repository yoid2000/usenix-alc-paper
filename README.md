# Code for paper submission to PETs 2025.3, paper 36

This code will be placed in a non-anonymous location upon acceptance.

Note that there is no easy install script for this code. The user must install all dependent software individually.

## alscore.py

This is a class that implements the PCC and ALC computations, given the base measures of Pbase, Cbase, Patk, and Catk. This file is suitable to import into any implementation of our method.

## compares.py

This code makes the comparison between our approach and the prior approaches of Giomi and Stadler. The results of this code are covered in Section 5. The code is designed to run on a cluster that uses SLURM to manage jobs.

To run, two environment variables should be created:

`ALC_TEST_DIR`: the path to a directory where the tests are run

`ALC_TEST_CODE`: the path to the directory containing the code (this directory)

(If not created, both default to the current directory)

The original datasets used in the experiments must be in directory `ALC_TEST_DIR/original_data_parquet` as parquet files. These can be found in this repo.

`python compares.ph make_syn` sets up the directories and files needed to generate the synthetic data files used in to make the comparisons. It places the required original datasets in `ALC_TEST_DIR/synDatasets`. It also creates the SLURM script `make_syn.slurm` and places it in `ALC_TEST_DIR`. The synthetic files are then created by running `sbatch make_syn.slurm`.

If a SLURM cluster is not available, each individual synthetic datafile can be created with `python compares.ph make_one_syn x`, where x is an integer ranging from 0 to the total number of synthetic datasets.

Note that SDV must be installed to make the synthetic datasets.

`python compares.py config` generates two configuration files and places them in `ALC_TEST_DIR`:

* `measure_jobs.json`: Contains the commands run by the SLURM sbatch jobs
* `measure.slurm`: Contains the SLURM script, executed with `sbatch measure.slurm`

The results of these jobs are placed in the directory `ALC_TEST_DIR/instances` as a set of `.json` files.

If a SLURM cluster is not available, the individual compare jobs can be run with `python compares.py measure x`, where x is an integer ranging from 0 to the total number of compare jobs.

Upon first run, `python compares.py plot` gathers the data in the `.json` files in `instances` and places it in the file `ALC_TEST_DIR/measure.parquet`. On subsequent runs, it reads the data out of `measure.parquet`. It then generates the plots used in the paper. The plots are placed in `ALC_TEST_DIR/plots`.

`basic_by_secret_col.png` corresponds to Figure 9 in the paper. `alc_improve_by_secret_percent_custom.png` corresponds to Figure 10 in the paper.

## plots.py

`python plots.py` generates the plots for Figures 4, 5, and 6 and places them in `ALC_TEST_DIR/als_plots`. `prec_cov_for_equal_pcc.png` corresponds to Figure 4. `prec_cov_for_diff_alpha.png` corresponds to Figure 5. `prec_cov_for_diff_cmin.png` corresponds to Figure 6.

## alc_plots.py

`python alc_plots.py` generates the plots for Figures 7 and 8 and places them in `ALC_TEST_DIR/alc_plots`. `alc_basic.png` corresponds to Figure 7. `alc_cov.png` corresponds to Figure 8.

## independence.py

`independence.py` runs the experiments used to measure the effect of dependent records (Figure 3).

The original datasets used in the experiments must be in directory `ALC_TEST_DIR/original_data_parquet` as parquet files. These can be found in this repo.

`python independence.py measure x` runs one measurement job, where x is an integer ranging from 0 to the number of measures. The resulting measures are placed in `.json` files in directory `ALC_TEST_DIR/independence_results`.

To run these jobs on a SLURM cluster, run the following SLURM script with `sbatch` (this is not automatically generated):

```
#!/bin/bash
#SBATCH --job-name=independences
#SBATCH --output=/INS/syndiffix/work/paul/alc/tests/slurm_independence_out/out.%a.out
#SBATCH --time=7-0
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-2280
arrayNum="${SLURM_ARRAY_TASK_ID}"
source /INS/syndiffix/work/paul/alc/tests/sdx_venv/bin/activate
python /INS/syndiffix/work/paul/alc/usenix-alc-paper/independence.py measure $arrayNum
```

`python independence.py gather` collects the results in the `.json` files in `independence_results`, and places them in `independence_results/independence_results.parquet`.

`python independence.py plot` reads in `independence_results.parquet`, generates plots, and places them in `ALC_TEST_DIR/independence_results`. `ind_default_overfit2.png` corresponds to Figure 3.
