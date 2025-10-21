# Import modules
import numpy as np
import pandas as pd
import time 

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
        "w" : 1.0,
        "particles":100,
        "iterations":100,
    }
    return params

def runPSO(params):
    # Set-up hyperparameters
    options = {'c1': params["c1"], 'c2': params["c2"], 'w':params["w"]}
    topology = ps.backend.topology.Star()

    # Call instance of PSO
    optimiser = ps.single.GeneralOptimizerPSO(n_particles=params["particles"], dimensions=ndim, bounds=bounds, options=options, topology=topology)

    # Perform optimization
    cost, pos = optimiser.optimize(fx.rastrigin, iters=params["iterations"],verbose=False)

    #print("Best solution: {}, fitness value: {}".format(pos,cost))
    return cost,optimiser.cost_history
    #print(optimiser.cost_history)

def runExperiment(params,logFile,label):
    # time the script
    start = time.time()

    nreps = 30
    best_reps = []
    best_its = {}
    for rep in range(nreps):
        bestFit,best_it = runPSO(params)
        #print("Best fitness of repetiton {} : {}".format(rep,bestFit))
        best_reps.append(bestFit)

        for it in range(len(best_it)):
             if not it in best_its:
                best_its[it] = []
             best_its[it].append(best_it[it])

    trace = []
    for it in sorted(best_its.keys()):
        trace.append([it,np.mean(best_its[it]),label])

    df = pd.DataFrame(trace)
    df.to_csv(logFile,header=False,index=False,mode="a")

    print("Average of best fitness for experiment {} : {}".format(label,np.mean(best_reps)))

    end = time.time()
    print("Average run time for experiment {} : {}".format(label,(end-start)/nreps))


outdir = "PSO_Logs"
import os
if not os.path.exists(outdir):
    os.mkdir(outdir)

# Create an empty data frame to initialise the CSV file 
df = pd.DataFrame(columns=["It","AveBestFitness","ExperimentName"])

# Create the parameters data structure
params = initParams()

# As an example, we will perform a parameter sweep for the probability of crossover
values_particles = [1,     2,    5,      10,  50, 100, 200, 500, 1000]
values_iterations =[10000, 5000, 2000, 1000, 200, 100,  50,  20,   10]
logName = "particle_it"
logFile = f"{outdir}/trace-{logName}.csv"
df.to_csv(logFile,index=False)

for particles, iterations in zip(values_particles, values_iterations):
    params["particles"] = particles
    params["iterations"] = iterations
    runExperiment(params,logFile,"particles={} | iterations={}".format(particles, iterations))


