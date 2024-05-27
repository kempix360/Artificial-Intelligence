from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 8

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm
    # 1. Selection
    sum_of_fitness = 0
    for individual in population:
        sum_of_fitness += fitness(items, knapsack_max_capacity, individual)
    selection_probabilities = [fitness(items, knapsack_max_capacity, individual) / sum_of_fitness
                               for individual in population]

    selected_population = random.choices(range(population_size), weights=selection_probabilities, k=n_selection)

    # 2. Crossover
    new_population = []
    size = int((population_size - n_elite)/2)
    for _ in range(size):
        parent_1 = population[random.choice(selected_population)]
        parent_2 = population[random.choice(selected_population)]
        crossover_point = int(len(items)/2)
        child_1 = parent_1[crossover_point:] + parent_2[:crossover_point]
        child_2 = parent_2[crossover_point:] + parent_1[:crossover_point]
        new_population.append(child_1)
        new_population.append(child_2)

    # 3. Mutation
    for i in range(len(new_population)):
        random_index = random.randint(0, len(new_population[i]) - 1)
        new_population[i][random_index] = not new_population[i][random_index]

    # 4. Elitism
    elite_population = sorted(population, key=lambda individual: fitness(items, knapsack_max_capacity, individual),
                          reverse=True)[:n_elite]
    for i in range(0, len(elite_population)):
        new_population.append(elite_population[i])
    population = new_population

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
