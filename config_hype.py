import numpy as np
from hype.backends import SLURMBackend
from hype.data import Objective
from hype.experiments import DefaultExperiment
from hype.formatters import RegExpLogFormatter
from hype.optimizers import RandomSearch

# Specify the tunable parameters as cmd arguments and their possible ranges
params_to_tune = {
    "--lr": {
      "type": "values", "values": [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.008, 0.01, 0.05] 
    },
    # "--weight_decay": {
    #   "type": "range", "min": 1e-6, "max": 0.1
    # },
    "--hidden_dim": {
      "type": "values", "values": [16, 32, 64, 128 ] 
    },
       "--num_hid_layers": {
      "type": "values", "values": [2, 3, 4] 
    },
    "--dropout": {
        "type": "values", "values": [0.3, 0.5, 0.8] 
    },
}

# Specify result pattern used to parse logs
formatter = RegExpLogFormatter(r'loss_train:\s(.*) acc_train', group=1)

# Maximize or minimize
objective = Objective.MINIMIZE

# BACKEND parameters. We will use SLURMBackend to run on DB Cluster
backend = SLURMBackend(
    script_to_run="/nfs/team/mlo/aswamy/code/gcn_node_classification/gcn_train_slurm.slurm",
    slurm_master="wood.int.europe.naverlabs.com",
    slurm_partition="gpu-mono",
    username="aswamy",  # CHANGE THIS
    ssh_key_path="/home/aswamy/.ssh/hype_rsa",  # CHANGE THIS
    num_workers=6
)

search_algorithm = RandomSearch(
    params_to_tune=params_to_tune,
    objective=objective,
)

experiment = DefaultExperiment(
    algorithm=search_algorithm,
    backend=backend, num_generations=287,
    objective=objective, formatter=formatter
)
