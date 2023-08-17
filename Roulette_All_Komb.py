import random
import math
import pandas as pd
import time
import numpy as np
import os


script_directory = os.path.dirname(os.path.abspath(__file__))
file_name = 'Roulette_results.xlsx'

file_path = os.path.join(script_directory, file_name)
writer = pd.ExcelWriter(file_path, engine='xlsxwriter')

#%% Probleminitialisierung
city_coordinates = {
    'A': (0, 0),
    'B': (10, 5),
    'C': (3, 15),
    'D': (12, 10),
    'E': (2, 9),
    'F': (6, 0),
    'G': (15, 2),
    'H': (7, 6),
    'I': (22, 1),
    'J': (1, 8),
    # 'K': (5, 12),
    # 'L': (18, 7),
    # 'M': (9, 3),
    # 'N': (4, 18),
    # 'O': (11, 9),
    # 'P': (20, 25),
    # 'Q': (30, 35),
    # 'R': (27, 28),
    # 'S': (17, 21),
    # 'T': (33, 19),
    # '21': (40, 12),
    # '22': (25, 30),
    # '23': (36, 28),
    # '24': (8, 22),
    # '25': (21, 37),
    # '26': (19, 11),
    # '27': (50, 50),
    # '28': (60, 45),
    # '29': (55, 60),
    # '30': (70, 80),
    # '31': (85, 75),
    # '32': (90, 95),
    # '33': (80, 90),
    # '34': (95, 85),
    # '35': (75, 70),
    # '36': (65, 65),
    # '37': (58, 42),
    # '38': (63, 47),
    # '39': (41, 60),
    # '40': (53, 58),
    # '41': (62, 61),
    # '42': (70, 62),
    # '43': (80, 65),
    # '44': (85, 58),
    # '45': (95, 60),
    # '46': (91, 75),
    # '47': (98, 71),
    # '48': (80, 80),
    # '49': (85, 89),
    # '50': (93, 94),
    # '51': (84, 98),
    # '52': (80, 97),
    # '53': (77, 88),
    # '54': (65, 90),
    # '55': (70, 95),
    # '56': (75, 82),
    # '57': (62, 90),
    # '58': (53, 85),
    # '59': (40, 90),
    # '60': (50, 88),
    # '61': (45, 78),
    # '62': (42, 66),
    # '63': (48, 65),
    # '64': (56, 78),
    # '65': (62, 72),
    # '66': (70, 70),
    # '67': (72, 78),
    # '68': (68, 85),
    # '69': (75, 82),
    # '70': (80, 78),
    # '71': (78, 70),
    # '72': (84, 72),
    # '73': (86, 75),
    # '74': (80, 77),
    # '75': (83, 68),
    # '76': (75, 63),
    # '77': (70, 68),
    # '78': (60, 70),
    # '79': (65, 75),
    # '80': (69, 78),
    # '81': (63, 85),
    # '82': (58, 80),
    # '83': (55, 78),
    # '84': (52, 70),
    # '85': (50, 75),
    # '86': (57, 72),
    # '87': (60, 78),
    # '88': (65, 80),
    # '89': (70, 77),
    # '90': (75, 75),
    # '91': (80, 80),
    # '92': (82, 85),
    # '93': (77, 90),
    # '94': (70, 88),
    # '95': (65, 84),
    # '96': (68, 79),
    # '97': (73, 81),
    # '98': (76, 85),
    # '99': (80, 83),
    # '100': (83, 80),
}


#%% Parameter
mutation_rates = [0, 0.02, 0.1, 0.5, 1]
population_sizes = [20 , 100, 1000]
num_runs = 10
num_generations = 50
elite_size = 1
starting_city = 'A'
tournament_selection_size = 2

# 50 Generationen für 10
# 150 Generationen für 150



#%% Initialpopulation
def generate_population(size):
    population = []
    for _ in range(size):
        chromosome = list(city_coordinates.keys())
        random.shuffle(chromosome)
        population.append(chromosome)
    return population

#%% Calculate Eucleadean Distance
def calculate_distance(city1, city2):
    x1, y1 = city_coordinates[city1]
    x2, y2 = city_coordinates[city2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#%% Fitnessevaluation, je kleiner desto besser
def calculate_fitness(chromosome):
    total_distance = 0
    num_cities = len(chromosome)
    for i in range(num_cities):
        from_city = chromosome[i]
        to_city = chromosome[(i + 1) % num_cities]
        total_distance += calculate_distance(from_city, to_city)
    return total_distance

#%% Linear order Crossover
def crossover(parent1, parent2):
    num_cities = len(parent1)
    child = [-1] * num_cities
    start, end = sorted([random.randint(0, num_cities - 1), random.randint(0, num_cities - 1)])
    child[start:end+1] = parent1[start:end+1]
    remaining_cities = [city for city in parent2 if city not in child]
    j = 0
    for i in range(num_cities):
        if child[i] == -1:
            child[i] = remaining_cities[j]
            j += 1
    return child

#%% Single Swap mutation
def swap_mutation(chromosome):
    for i in range(len(chromosome)):
        if random.random() < mutation_rates:
            j = random.randint(0, len(chromosome) - 1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

#%% Inversion Mutation
def mutate(chromosome, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i:j+1] = reversed(chromosome[i:j+1])
    return chromosome

#%% Roulette selection
def roulette_wheel_selection(population):
    ranked_population = sorted (population, key =calculate_fitness)
    fitness_sum = sum(1/ (rank+1)for rank in range (len(ranked_population)))
    selection_probabilities =[(1/(rank+1)) / fitness_sum for rank in range(len(ranked_population))]
    selected=random.choices(ranked_population,weights=selection_probabilities,k=len(population)-elite_size)
    selected.extend(sorted(population, key=calculate_fitness)[:elite_size])
    return selected


#%% Hauptalgo
def genetic_algorithm(mutation_rate, population_size):
    computing_times = []
    best_distances_per_run = []  # Neue Variable -> beste Distanz pro Run und nicht pro Generation

    for run in range(num_runs):
        start_time = time.time()
        best_distances = []  # Für jeden neuen Run best distances = clear, da best_distances alle Distanzen der gesamten Generationen trackt! (war für Plotten am Anfang wichtig...)
        population = generate_population(population_size)

        for generation in range(num_generations):
            population = roulette_wheel_selection(population)


            next_generation = []
            next_generation.extend(sorted(population, key=calculate_fitness)[:elite_size])

            while len(next_generation) < population_size:
                parent1, parent2 = random.sample(population, 2)
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                next_generation.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

            population = next_generation[:population_size]

            best_chromosome = min(population, key=calculate_fitness)
            best_distance = calculate_fitness(best_chromosome)
            best_distances.append(best_distance)

        # Computing Time
        end_time = time.time()
        computing_time = end_time - start_time
        computing_times.append(computing_time)

        # Distances
        best_distances_per_run.append(best_distances[-1])  # Fügt die beste Distanz der letzten Generation hinzu

    # Berechne Durchschnitt und Standardabweichung der besten Distanzen pro Run
    average_best_distance = sum(best_distances_per_run) / len(best_distances_per_run)
    std_deviation_best_distance = np.std(best_distances_per_run)

    # Berechne Durchschnitt der Computing Times
    average_computing_time = sum(computing_times) / len(computing_times)

    # Erstelle ein DataFrame mit der durchschnittlichen besten Distanz, der durchschnittlichen Computing Time und der Standardabweichung der besten Distanzen pro Run
    df = pd.DataFrame({'Distance': average_best_distance,
                       'Computing Time': average_computing_time,
                       'Std.Dev. Distance': std_deviation_best_distance}, index=[0])

    return df



# Erstelle ein leeres DataFrame, um die Ergebnisse für alle Parameterkombinationen zu speichern
all_results = pd.DataFrame()

for mutation_rate in mutation_rates:
    for population_size in population_sizes:
        # Führe den genetischen Algorithmus mit den aktuellen Parametern aus
        # und erhalte die Ergebnisse
        results = genetic_algorithm(mutation_rate, population_size)

        # Erstelle eine neue Zeile im DataFrame mit den Parameterwerten und den Ergebnissen
        results['Mutation Rate'] = mutation_rate
        results['Population Size'] = population_size

        # Füge die Ergebnisse zur Gesamtliste hinzu
        all_results = pd.concat([all_results, results])

columns_order = ['Mutation Rate', 'Population Size', 'Distance', 'Computing Time', 'Std.Dev. Distance']
all_results = all_results[columns_order]

# Schreibe das kombinierte DataFrame in ein Tabellenblatt der Excel-Datei
all_results.to_excel(writer, sheet_name='Results', index=False)

# Speichere die Excel-Datei
writer.save()

'''


1 0 0
2 10 5
3 3 15
4 12 10
5 2 9
6 6 0
7 15 2
8 7 6
9 22 1
10 1 8
11 5 12
12 18 7
13 9 3
14 4 18
15 11 9
16 20 25
17 30 35
18 27 28
19 17 21
20 33 19

'''