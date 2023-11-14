###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Onim Sarker       			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from torch import initial_seed
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller


# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import random
from deap import base, creator
from deap import tools


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'individual_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[3],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  randomini = "yes")

#You may notice that i defined env again in this function. This is so we can used the last function in this code called startUpOnePoint()
def deap_specialist_cxOnePoint(experiment_name, enemyNumber,iterationnumber):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    if os.path.exists(experiment_name + '/results.csv'):
        os.remove(experiment_name + '/results.csv')

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(
        experiment_name=experiment_name,
        enemies=[enemyNumber],
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        randomini = "yes"
    )
    
    main()

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons, I used this in Main 
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


def fnInitialize(iPopSize):
    "initialize objects in DEAP"
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=iPopSize)
    toolbox.register(
            "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n_vars
        )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #Evaluate the operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate",  tools.mutFlipBit, indpb=0.9)
    #toolbox.register("mutate",  tools.mutGaussian, mu = 0, sigma = 1)
    toolbox.register("select", tools.selTournament, tournsize = 3)
    toolbox.register("evaluate", evaluate)
    toolbox.register("survive", tools.selBest)
    return toolbox
    
    


# runs simulation
def simulation(env,x):
    #I don't really use this. I think we can use it if we want to se it visually
    f,p,e,t = env.play(pcont=x)
    return f


def evaluate(pop):
    """
    This function will start a game with one individual from the population

    Args:
        individual (np.ndarray of Floats between -1 and 1): One individual from the population

    Returns:
        Float: Fitness
    """
    vTotalFitness = [] 
    for ind in pop:
        indArray = np.array(ind) 
        f, p, e, t = env.play(pcont=indArray)  # return fitness, self.player.life, self.enemy.life, self.time
        vTotalFitness.append(f)
        ind.fitness.values = [f]
    return vTotalFitness




def main():
    
    #Intial Parameter
    iPopSize = 20
    iNumberTotalGen = 15
    dMutProb = 0.90
    dCrossoverProb = 0.9
    
    toolbox = fnInitialize((env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5)
    pop = toolbox.population(iPopSize)

    vTotFit = toolbox.evaluate(pop)
    print(vTotFit)
    for ind, fit in zip(pop, vTotFit):   
        ind.fitness.values = [fit]
    print("DONE INITIAL UPDATE POPULATION")
    vGenMean = [] 
    vGenMax = [] 
    
        
    #Mating need to figure out a way to keep the population size over the generations the same
    for iG in range(iNumberTotalGen):
        print("Generation Number : ", iG +1)
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        dHighestFitness = 0 
        #Crossover Need to think about a way to make the probablity of mating high for two individuals with a high fitness
        children = []
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < dCrossoverProb:
                toolbox.mate(child1, child2)
                del child1.fitness.values 
                del child2.fitness.values
                #children.extend((child1,child2))
                
        print("DONE CROSSOVER")
        #offspring.extend((child for child in children))
        #Mutation
        for mutant in children:
            if random.random() < dMutProb:
                print("MUTATING")
                toolbox.mutate(mutant)
                del mutant.fitness.values
        print("DONE MUTATING")
        offspring.extend((child for child in children))
        
    
        #Calculate fitness new generation
        vChangedInd = [ind for ind in offspring if not ind.fitness.valid]
        
        #Evaluate new popoulation 
        vTotalFitness = toolbox.evaluate(vChangedInd)
        
        for ind, fit in zip(vChangedInd, vTotalFitness):   
            ind.fitness.values = [fit]
        print("DONE LAST UPDATE POPULATION")
        survivors = toolbox.survive(offspring, iPopSize)
        #survivors = toolbox.survive(offspring, POP_SIZE)
        #print(vChangedInd.fitness[0])
        pop[:] = survivors
        vGenFitness = toolbox.evaluate(pop)
        print("MAX FITNESS :",np.amax(vGenFitness), '  Location :', np.argmax(vGenFitness))
        if np.amax(vGenFitness) >= dHighestFitness:
            dHighestFitness = np.amax(vGenFitness)
            vBestGene = pop[np.argmax(vGenFitness)]
        vGenMean.append(np.mean(vGenFitness))
        vGenMax.append(np.max(vGenFitness))
    vFinalFitness = toolbox.evaluate(pop)
    vBestIndividual = pop[np.argmax(vGenFitness)]
    
    print("************************** BEST INDIVIDUAL GENES*******************************")
    print(vBestGene)
    print("*******************************************************************************")
    #print(vFinalFitness)
    #print(vGenMean)
    #print(vGenMax)
    return(vFinalFitness, vGenMean, vGenMax)


def fnTrials():
    mFinalFitness = [] 
    mGenMean = [] 
    mGenMax = [] 
    for i in range(1, 11):
        vFinalFitness, vGenMean, vGenMax = main()
        mFinalFitness.append(vFinalFitness)
        mGenMean.append(vGenMean)
        mGenMax.append(vGenMax)
    


def startUpOnePoint():
    print("------------------------------- START ONEPOINT -------------------------------------------------------")
    # --------- STARTS PROGRAM FOR EVERY ENEMY 10 TIMES ---------------
    iterationnumber = 0
    for x in range(2,5):
        print("-----------ONEPOINT------------ ENEMY " + str(x) + " -------------------------------------------------------")
        enemyNumber = x
        experiment_name = "deap_specialist_cxOnePoint/Enemy" + str(enemyNumber)

        for i in range(1,11):
            iterationnumber = iterationnumber + 1
            print("----------ONEPOINT------------ RUN " + str(i) + " ----------- ENEMY " + str(x) + "--------------------------")
            experiment_name_temp = experiment_name + "/" + str(i)
            deap_specialist_cxOnePoint(experiment_name_temp, enemyNumber,iterationnumber)