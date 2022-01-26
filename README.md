# Resource-rational nudging

Pre-registrations, experiment demos, modeling, and analysis from Mouselab nudging experiments.

## Pre-registrations

* Experiment 1: https://osf.io/4fcny
* Experiment 2: https://osf.io/8vnxj
* Experiment 3: https://osf.io/8pyex
* Experiment 4: https://osf.io/cs5kx
* Experiment 5: https://osf.io/gspr4

## Experiment demos

Perform the experiments as they were given to participants at the links below.

* Experiment 1: https://default-options.netlify.app
* Experiment 2: https://suggested-alternatives.netlify.app
* Experiment 3: https://stoplight-highlighting.netlify.app
* Experiment 4: https://optimal-belief-modification.netlify.app
* Experiment 5: https://optimal-cost-modification.netlify.app

## Data

All reported experiments can be found in data/final_experiments_data. Additional pilot data can also be found in data/.

* Experiment 1: [default_data.csv](data/experiments/reported_experiments/default_data.csv)
* Experiment 2: [supersize_data.csv](data/experiments/reported_experiments/supersize_data.csv)
* Experiment 3: [stoplight_data.csv](data/experiments/reported_experiments/stoplight_data.csv)
* Experiment 4: [optimal_nudging_changing_belief_state_data.csv](data/experiments/reported_experiments/optimal_nudging_changing_belief_state_data.csv)
* Experiment 5: [optimal_nudging_changing_costs_data.csv](data/experiments/reported_experiments/optimal_nudging_changing_costs_data.csv)

## Model simulations

In the model/ directory, running `julia -p auto main.jl` will generate all the model simulations in model/results. 
Install all dependencies by running `julia install_deps.jl`.
It should run with any 1.x version of Julia.
Note that the naming conventions for the different experiments have changed over time. Sorry.

* Experiment 1: [default_options.jl](model/default_options.jl)
* Experiment 2: [supersize.jl](model/supersize.jl)
* Experiment 3: [stoplight.jl](model/stoplight.jl)
* Experiments 4 & 5: [optimal_nudging.jl](model/optimal_nudging.jl)

## Analyses

In the analysis/ directory, running `make results` will generate all the figures and statistics in the paper. Stats are generated as TeX files in analysis/stats.

* Experiment 1: [default.r](analysis/default.r)
* Experiment 2: [supersize.r](analysis/supersize.r)
* Experiment 3: [stoplight.r](analysis/stoplight.r)
* Experiment 4: [belief_modification.r](analysis/belief_modification.r)
* Experiment 5: [cost_rehttps://princeton.zoom.us/my/fredcallawayduction.r](analysis/cost_reduction.r)

