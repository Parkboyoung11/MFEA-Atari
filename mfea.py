from atari_benchmark import *
from slgep_lib import *
from mfea_lib import *
from utils import Evaluator

from joblib import Parallel, delayed
from os import cpu_count
from tqdm import trange
from time import time

def mfea(config, callback):
    # Problem
    taskset = Taskset(config)

    # Model
    cf = ChromosomeFactory(config)

    # Simple parameter
    K = taskset.K
    config['K'] = K
    N = config['pop_size'] * K
    T = config['num_iter']
    mutation_rate = config['mutation_rate']
    rmp = config['rmp']

    # Initialization
    population   = cf.initialize()
    skill_factor = np.array([i % K for i in range(2 * N)])
    factorial_cost = np.full([2 * N, K], np.inf)
    scalar_fitness = np.empty([2 * N])

    # For parallel evaluation
    print('[+] Initializing evaluators')
    evaluators = [Evaluator(config) for _ in range(2 * N)]
    stabe_evaluators = [Evaluator(config) for _ in range(30)]

    # First evaluation (sequential)
    delayed_functions = []
    for i in range(2 * N):
        sf = skill_factor[i]
        delayed_functions.append(delayed(evaluators[i].evaluate)(population[i], sf))
    fitnesses = Parallel(n_jobs=cpu_count())(delayed_functions)
    for i in range(2 * N):
        sf = skill_factor[i]
        factorial_cost[i, sf] = fitnesses[i]
    scalar_fitness = calculate_scalar_fitness(factorial_cost)

    # Evolve
    iterator = trange(T)
    for t in iterator:
        # permute current population
        permutation_index = np.random.permutation(N)
        population[:N] = population[:N][permutation_index]
        skill_factor[:N] = skill_factor[:N][permutation_index]
        factorial_cost[:N] = factorial_cost[:N][permutation_index]
        factorial_cost[N:] = np.inf

        # select pair to crossover
        for i in range(0, N, 2):
            # extract parent
            p1, p2 = population[i], population[i + 1]
            sf1, sf2 = skill_factor[i], skill_factor[i + 1]

            # recombine parent
            if sf1 == sf2:
                c1, c2 = cf.one_point_crossover_adf(p1, p2)
                c1 = cf.uniform_mutate(c1, mutation_rate)
                c2 = cf.uniform_mutate(c2, mutation_rate)
                skill_factor[N + i] = sf1
                skill_factor[N + i + 1] = sf1
            elif sf1 != sf2 and np.random.rand() < rmp:
                c1, c2 = cf.one_point_crossover_adf_multitask(p1, p2)
                c1 = cf.uniform_mutate(c1, mutation_rate)
                c2 = cf.uniform_mutate(c2, mutation_rate)
                if np.random.rand() < 0.5: skill_factor[N + i] = sf1
                else: skill_factor[N + i] = sf2
                if np.random.rand() < 0.5: skill_factor[N + i + 1] = sf1
                else: skill_factor[N + i + 1] = sf2
            else:
                p2  = find_relative(population, skill_factor, sf1, N)
                c1, c2 = cf.one_point_crossover_adf(p1, p2)
                c1 = cf.uniform_mutate(c1, mutation_rate)
                c2 = cf.uniform_mutate(c2, mutation_rate)
                skill_factor[N + i] = sf1
                skill_factor[N + i + 1] = sf1
            population[N + i, :], population[N + i + 1, :] = c1[:], c2[:]
        # evaluation
        delayed_functions = []
        for i in range(2 * N):
            sf = skill_factor[i]
            delayed_functions.append(delayed(evaluators[i].evaluate)(population[i], sf))
        fitnesses = Parallel(n_jobs=cpu_count())(delayed_functions)
        for i in range(2 * N):
            sf = skill_factor[i]
            factorial_cost[i, sf] = fitnesses[i]
        scalar_fitness = calculate_scalar_fitness(factorial_cost)

        # sort
        sort_index     = np.argsort(scalar_fitness)[::-1]
        population     = population[sort_index]
        skill_factor   = skill_factor[sort_index]
        factorial_cost = factorial_cost[sort_index]
        scalar_fitness = scalar_fitness[sort_index]

        # optimization info
        message = {'algorithm': 'mfea', 'instance':config['name']}
        results = get_optimization_results(t, population, factorial_cost, scalar_fitness, skill_factor, message)
        stabe_results = deepcopy(results)
        for i in range(len(stabe_results)):
            sf = stabe_results[i].sf
            chromosome = stabe_results[i].x
            stabe_delayed_functions = []
            for j in range(30):
                stabe_delayed_functions.append(delayed(stabe_evaluators[j].evaluate)(chromosome, sf))
            stabe_fitness = Parallel(n_jobs=cpu_count())(stabe_delayed_functions)
            stabe_results[i].fun = sum(stabe_fitness) / len(stabe_fitness)
        if callback:
            callback(results, stabe_results)
        desc = 'gen:{} fitness:{} message:{} K:{}'.format(t, ' '.join('{:0.2f}|{:0.2f}|{:0.2f}'.format(res.fun, res.mean, res.std) for res in results), message, K)
        iterator.set_description(desc)

