import itertools
import os
import random
import statistics
import scipy.stats
import time
from deap import creator, base, tools, algorithms, benchmarks
import numpy as np
import pandas as pd

# A positive weight indicates a maximisation function (higher is better), a negative weight a minimisation function (lower is better)
# The settings in the next block of lines define the problem to solve, so you cannot modify them
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_double", random.uniform, -5.12, 5.12)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_double, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", benchmarks.rastrigin)

def initParams():
    params = {
        "popSize":100,
        "iterations":100,
        "pcross":0.6,
        "pmut_ind":0.3,
        "pmut_gene":0.05,
        "tournsize":3,
        "mu":0,
        "sigma":0.5,
        "selection":tools.selTournament,
        "crossover":tools.cxOnePoint,
        "mutation":tools.mutGaussian
    }
    return params

def runGA(params):
    # These define the operators of the GA, and can be selected/tuned for the problem at hand
    toolbox.register("mate", params["crossover"])
    toolbox.register("mutate", params["mutation"], mu=params["mu"], sigma=params["sigma"], indpb=params["pmut_gene"])
    toolbox.register("select", params["selection"], tournsize=params["tournsize"])

    # This is the population size, the number of alternative solutions being evolved
    popSize=params["popSize"]
    population = toolbox.population(n=popSize)

    hof = tools.HallOfFame(1)

    # Evaluate the initial population
    fits = toolbox.map(toolbox.evaluate, population)
    for fit, ind in zip(fits, population):
        ind.fitness.values = fit

    hof.update(population)

    best_it = []
    # convergence = []
    # confidence_interval = []
    # Number of iterations of the evolutionary cycle
    NGEN=params["iterations"]
    for gen in range(NGEN):
        population = toolbox.select(population, k=len(population))
        offspring = algorithms.varAnd(population, toolbox, cxpb=params["pcross"], mutpb=params["pmut_ind"])

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        hof.update(population)
        top1 = tools.selBest(population, k=1)

        best_it.append(top1[0].fitness.values[0])

        # current_convergence = determine_convergence(population)
        # convergence.append(current_convergence)
        #
        # current_confidence_interval = determine_confidence_interval(population)
        # confidence_interval.append(current_confidence_interval)

    best = hof[0]
    #print("Best individual {} -> fitness: {}".format(best,best.fitness.values[0]))
    return best.fitness.values[0],best_it #,convergence,confidence_interval

def determine_convergence(population):
    return 0
    dimension_count = len(population[0][:])

    # compute the average distance from the mean position - std deviation - in each dimension
    std_deviations = []
    for dimension in range(dimension_count):
        positions = [coordinates[dimension] for coordinates in population]
        std_deviations.append(statistics.stdev(positions))
    return statistics.mean(std_deviations)

def determine_confidence_interval(population, confidence=0.95):
    return 0
    a = 1.0 * np.array([ind.fitness.values[0] for ind in population])
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


# noinspection DuplicatedCode
def runExperiment(params,logFile,label):
    # time the script
    start = time.time()

    nreps = 30
    best_reps = []
    best_its = {}
    # convergences = {}
    # confidence_intervals = {}

    for rep in range(nreps):
        # bestFit,best_it,convergence,confidence_interval = runGA(params)
        bestFit,best_it = runGA(params)
        #print("Best fitness of repetiton {} : {}".format(rep,bestFit))
        best_reps.append(bestFit)

        for it in range(len(best_it)):
            best_its.setdefault(it, [])
            best_its[it].append(best_it[it])

            # convergences.setdefault(it, [])
            # convergences[it].append(convergence[it])
            #
            # confidence_intervals.setdefault(it, [])
            # confidence_intervals[it].append(confidence_interval[it])

    trace = []
    for it in sorted(best_its.keys()):
        # trace.append([it,np.mean(best_its[it]),np.mean(convergences[it]),np.mean(confidence_intervals[it]),label])
        trace.append([it,np.mean(best_its[it]),label])

    df = pd.DataFrame(trace)
    df.to_csv(logFile,header=False,index=False,mode="a")

    print("Average of best fitness for experiment {} : {}".format(label,np.mean(best_reps)))

    end = time.time()
    print("Average run time for experiment {} : {}".format(label,(end-start)/nreps))


# Create an empty data frame to initialise the CSV file
# by Convergence we mean the standard deviation of all points in each dimensions. The average value of each dimensional std. dev is taken for a single convergence number
# df = pd.DataFrame(columns=["It","AveBestFitness","AvgConvergence", "ConfidenceInterval", "ExperimentName"])
df = pd.DataFrame(columns=["It","AveBestFitness", "ExperimentName"])
logName = "sigma"
outdir = "GA_Second-Sweep_Logs"
logFile = f"{outdir}/trace-{logName}.csv"
if not os.path.exists(outdir):
    os.mkdir(outdir)
df.to_csv(logFile,index=False)

# Create the parameters data structure
params = initParams()

# As an example, we will perform a parameter sweep for the probability of crossover
popsize_iterations_vals = [(100000, 1), (5000, 2), (2500, 4), (1000, 10), (500, 20), (250, 40), (100, 100), (50, 200), (25, 400), (10, 1000)]
pcross_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
pmut_ind_vals = [0.01, 0.1, 0.3, .5, .6, .7, 7.5, .8, .85, .9, .95, .99]
pmut_gene_vals = [.001, .0025, .005, .01, .025, .05, .1, .25, .5]
tournsize_vals = [1, 2, 3, 5, 10, 25, 50, 75, 90]
sigma_vals = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]

for sigma in sigma_vals:
    params["sigma"] = sigma
    runExperiment(params,logFile,"default".format())

