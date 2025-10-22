# Import modules
import numpy as np
import pandas as pd
import time
import random
from itertools import product

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

# Definition of the problem bounds, do not touch this block of code
ndim = 10
max_bound = 5.12 * np.ones(ndim)
min_bound = - max_bound
bounds = (min_bound, max_bound)


def initParams():
    params = {
        "c1": 1.0,
        "c2": 1.0,
        "w" : 0.75,
        "particles":100,
        "iterations":100,
        "topology": ps.backend.topology.Star()
    }
    return params

def runPSO(params):
    # Set-up hyperparameters
    options = {key: params[key] for key in params if key not in ['particles', "iterations", "topology"]}
    # options = {'c1': params["c1"], 'c2': params["c2"], 'w':params["w"]}
    topology = params["topology"]

    # Call instance of PSO
    optimiser = ps.single.GeneralOptimizerPSO(n_particles=params["particles"], dimensions=ndim, bounds=bounds, options=options, topology=topology)

    # Perform optimization
    cost, pos = optimiser.optimize(fx.rastrigin, iters=params["iterations"],verbose=False)

    #print("Best solution: {}, fitness value: {}".format(pos,cost))
    convergence_history = [determine_convergence(pos) for pos in optimiser.pos_history]

    return cost,optimiser.cost_history,convergence_history
    #print(optimiser.cost_history)

def determine_convergence(pos_history):
    return np.mean(np.std(pos_history, axis=0))

def runExperiment(params,logFile,label):
    # time the script
    start = time.time()

    nreps = 100
    best_reps = []
    best_its = {}
    convergence_its = {}

    for rep in range(nreps):
        # Set the random seed to ensure reproducability
        random.seed(rep)

        bestFit,best_it,conv_history = runPSO(params)
        #print("Best fitness of repetiton {} : {}".format(rep,bestFit))
        best_reps.append(bestFit)

        for it in range(len(best_it)):
            if not it in best_its:
                best_its[it] = []
                best_its[it].append(best_it[it])

            if not it in convergence_its:
                convergence_its[it] = []
            convergence_its[it].append(conv_history[it])

    trace = []
    for it in sorted(best_its.keys()):
        trace.append([it,np.mean(best_its[it]), np.mean(convergence_its[it]),label])

    df = pd.DataFrame(trace)
    df.to_csv(logFile,header=False,index=False,mode="a")

    print("Average of best fitness for experiment {} : {}".format(label,np.mean(best_reps)))

    end = time.time()
    print("Average run time for experiment {} : {}".format(label,(end-start)/nreps))


outdir = "PSO_Logs_2D_Sweeps"
import os
if not os.path.exists(outdir):
    os.mkdir(outdir)

def create_experiments(log_name):
    print(f" ============= STARTING EXPERIMENTS FOR {log_name} ============= ")
    # Create the parameters data structure
    params = initParams()
    df = pd.DataFrame(columns=["It", "AveBestFitness", "Convergence", "ExperimentName"])
    logFile = f"{outdir}/trace-{log_name}.csv"
    df.to_csv(logFile, index=False)

    return params, logFile
# As an example, we will perform a parameter sweep for the probability of crossover
values_particle_it = [
    # {'iterations': 5000, 'particles': 2},
    # {'iterations': 2000, 'particles': 5},
    # {'iterations': 1000, 'particles': 10},
    {'iterations': 200, 'particles': 50},
    {'iterations': 100, 'particles': 100},
    {'iterations': 50, 'particles': 200},
    {'iterations': 20, 'particles': 500},
    {'iterations': 10, 'particles': 1000}
]
values_c1 = [
    {'c1': 0.125},
    {'c1': 0.25},
    {'c1': 0.5},
    {'c1': 0.75},
    {'c1': 1},
    {'c1': 2},
    {'c1': 3},
    {'c1': 4}
]
values_c2 = [
    {'c2': 0.125},
    {'c2': 0.25},
    {'c2': 0.5},
    {'c2': 0.75},
    {'c2': 1},
    {'c2': 2},
    {'c2': 3},
    {'c2': 4}
]
values_w = [
    {'w': 0.125},
    {'w': 0.25},
    {'w': 0.5},
    {'w': 0.75},
    {'w': 1},
    {'w': 2},
    {'w': 3},
    {'w': 4}
]
values_topologies = [
    {"name": "Ring | k=1 | p=1", "topology": ps.backend.topology.Ring(), "k": 1, "p": 1},
    {"name": "Ring | k=2 | p=1", "topology": ps.backend.topology.Ring(), "k": 2, "p": 1},
    {"name": "Ring | k=5 | p=1", "topology": ps.backend.topology.Ring(), "k": 5, "p": 1},
    {"name": "Ring | k=10 | p=1", "topology": ps.backend.topology.Ring(), "k": 10, "p": 2},
    {"name": "Ring | k=1 | p=2", "topology": ps.backend.topology.Ring(),"k": 1, "p": 2},
    {"name": "Ring | k=2 | p=2", "topology": ps.backend.topology.Ring(), "k": 2, "p": 2},
    {"name": "Ring | k=5 | p=2", "topology": ps.backend.topology.Ring(), "k": 5, "p": 2},
    {"name": "Ring | k=10 | p=2", "topology": ps.backend.topology.Ring(), "k": 10, "p": 2},
    # {"name": "VonNeumann | p=1 | r=1", "topology": ps.backend.topology.VonNeumann(), "p": 1, "r": 1},
    # {"name": "VonNeumann | p=2 | r=1", "topology": ps.backend.topology.VonNeumann(), "p": 2, "r": 1},
]

params, logFile = create_experiments("c1-c2")
for args1, args2 in product(values_c1, values_c2):
    params = initParams()
    combined_args = {}
    combined_args.update(args1)
    combined_args.update(args2)
    params.update(combined_args)
    runExperiment(params, logFile, " | ".join([f"{key}={combined_args[key]}" for key in combined_args]))

params, logFile = create_experiments("particle_it-topology")
for args1, args2 in product(values_particle_it, values_topologies):
    params = initParams()
    combined_args = {}
    combined_args.update(args1)
    combined_args.update(args2)
    params.update(combined_args)
    runExperiment(params, logFile, " | ".join([f"{key}={combined_args[key]}" for key in ["name", *args1.keys()]]))

# # Run experiments for differing particle/iterations combos
# params, logFile = create_experiments("particle_it")
# for args in values_particle_it:
#     params = initParams()
#     params.update(args)
#     runExperiment(params, logFile, " | ".join([f"{key}={args[key]}" for key in args]))
#
# # Run experiments for c1 vals
# params, logFile = create_experiments("c1")
# for args in values_c1:
#     params = initParams()
#     params.update(args)
#     runExperiment(params, logFile, " | ".join([f"{key}={args[key]}" for key in args]))
#
# # Run experiments for c2 vals
# params, logFile = create_experiments("c2")
# for args in values_c2:
#     params = initParams()
#     params.update(args)
#     runExperiment(params, logFile, " | ".join([f"{key}={args[key]}" for key in args]))

# # Run experiments for w vals
# params, logFile = create_experiments("w")
# for args in values_w:
#     params = initParams()
#     params.update(args)
#     runExperiment(params, logFile, " | ".join([f"{key}={args[key]}" for key in args]))


# # Run experiments for differing topologies
# params, logFile = create_experiments("topologies")
# for args in values_topologies:
#     params = initParams()
#     params.update(args)
#     runExperiment(params, logFile, "topology={}".format(params["name"]))


# params, logFile = create_experiments("baseline")
# runExperiment(params, logFile, "baseline")




