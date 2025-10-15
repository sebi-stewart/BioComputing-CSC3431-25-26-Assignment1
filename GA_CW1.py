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
    convergence = []
    confidence_interval = []
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

        current_convergence = determine_convergence(population)
        convergence.append(current_convergence)

        current_confidence_interval = determine_confidence_interval(population)
        confidence_interval.append(current_confidence_interval)

    best = hof[0]
    #print("Best individual {} -> fitness: {}".format(best,best.fitness.values[0]))
    return best.fitness.values[0],best_it,convergence,confidence_interval

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


def runExperiment(params,logFile,label):
    # time the script
    start = time.time()

    nreps = 30
    best_reps = []
    best_its = {}
    convergences = {}
    confidence_intervals = {}

    for rep in range(nreps):
        bestFit,best_it,convergence,confidence_interval = runGA(params)
        #print("Best fitness of repetiton {} : {}".format(rep,bestFit))
        best_reps.append(bestFit)

        for it in range(len(best_it)):
            best_its.setdefault(it, [])
            best_its[it].append(best_it[it])

            convergences.setdefault(it, [])
            convergences[it].append(convergence[it])

            confidence_intervals.setdefault(it, [])
            confidence_intervals[it].append(confidence_interval[it])

    trace = []
    for it in sorted(best_its.keys()):
        trace.append([it,np.mean(best_its[it]),np.mean(convergences[it]),np.mean(confidence_intervals[it]),label])

    df = pd.DataFrame(trace)
    df.to_csv(logFile,header=False,index=False,mode="a")

    print("Average of best fitness for experiment {} : {}".format(label,np.mean(best_reps)))

    end = time.time()
    print("Average run time for experiment {} : {}".format(label,(end-start)/nreps))


# Create an empty data frame to initialise the CSV file
# by Convergence we mean the standard deviation of all points in each dimensions. The average value of each dimensional std. dev is taken for a single convergence number
df = pd.DataFrame(columns=["It","AveBestFitness","AvgConvergence", "ConfidenceInterval", "ExperimentName"])
logName = "brute_force_sweep"
outdir = "GA_log"
logFile = f"{outdir}/trace-{logName}.csv"
if not os.path.exists(outdir):
    os.mkdir(outdir)
df.to_csv(logFile,index=False)

# Create the parameters data structure
params = initParams()

# As an example, we will perform a parameter sweep for the probability of crossover
pcross = [0.5, 0.6, 0.7, 0.8, 0.9]
pmut_ind = [.5, .6, .7, .8, .85, .9, .95, .99]
pmut_gene = [.005, .025, .05, .1, .25]

pmut_intersect_experiments = itertools.product((.3, .8, .9, .95), (.05, .1, .25, ))

brute_force_sweep = itertools.product(pcross, pmut_ind, pmut_gene)
# print(len(list(brute_force_sweep)))
for pcross_prob, pmut_ind_prob, pmut_gene_prob in brute_force_sweep:
    params["pcross"] = pcross_prob
    params["pmut_ind"] = pmut_ind_prob
    params["pmut_gene"] = pmut_gene_prob
    runExperiment(params,logFile,"pcross={} | pmut_ind={} | pmut_gene{}".format(pcross_prob, pmut_ind_prob, pmut_gene_prob))

