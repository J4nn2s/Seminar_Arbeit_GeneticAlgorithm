import random
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from decimal import Decimal, getcontext

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

#kürzeste Distanzen
#67.4555 
#133.099
#561.841

#%% Parameter
population_size = 20 #Wie groÃŸ die population ursprÃ¼nglich --> 50 AusgangslÃ¶sungen
mutation_rate = 0.5
num_generations = 150 #Abbruchkriterium!
elite_size = 1
starting_city = 'A'
tournament_selection_size = 2

#%% Initialpopulation
def generate_population(size):
    population = []
    for _ in range(size):
        chromosome = list(city_coordinates.keys()) #macht aus dem Dictonary eine Liste mit Staedten [A,B,C...]
        random.shuffle(chromosome) #Sortiert die Liste ZUFAELLIG um --> RANDOM Approach
        population.append(chromosome) #Erstellt so insg. Pop_SIZE viele Chromosomes
    return population

#%% Calculate Eucleadean Distance
def calculate_distance(city1, city2):
    x1, y1 = city_coordinates[city1] #Ruft Dictonary auf und fuer jede Stadt (z.B. "A") x1= x Korrdinate y1=y Koordinate
    x2, y2 = city_coordinates[city2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) ##standardabweichung Wurzel aus (x2-x1)^2 + (y2-y1)^2

#%% Fitnessevaluation, je kleiner desto besser
def calculate_fitness(chromosome): 
    total_distance = 0
    num_cities = len(chromosome)
    for i in range(num_cities):
        from_city = chromosome[i] #Erste Stadt, am Anfang Stadt des Chromosomes an der Stelle 0
        to_city = chromosome[(i + 1) % num_cities] #Die naechste Stadt (i+1)
        total_distance += calculate_distance(from_city, to_city)#Total Distance + Distance zwischen Stadt i und Stadt (i+1)
    return total_distance

#%% Linear order Crossover
def crossover(parent1, parent2): #zufaellig ausgewaehlte sample von Elternteil 1 (z.B. 3-6), rest wird mit parent 2 aufgefuellt
    num_cities = len(parent1) 
    child = [-1] * num_cities 
    start, end = sorted([random.randint(0, num_cities - 1), random.randint(0, num_cities - 1)]) #randomly select Start / End Index e.g. 3-6
    child[start:end+1] = parent1[start:end+1] #copied corresponding segment e.g. 3-6 to child
    remaining_cities = [city for city in parent2 if city not in child] # adding remaining cities, that are not in the segment to the child from other parent
    j = 0
    for i in range(num_cities):
        if child[i] == -1: #wenn Child noch nicht ueberschrieben, also immer noch -1 --> ueberschreibe diese Stelle mit City von parent2
            child[i] = remaining_cities[j]
            j += 1
    return child

#%%Inversion Mutation
def mutate(chromosome): 
    if random.random() < mutation_rate:
        # Select two random indices
        i, j = random.sample(range(len(chromosome)), 2)
        # Invert the order of the cities between the two indices
        chromosome[i:j+1] = reversed(chromosome[i:j+1])
    return chromosome

def tournament_selection(population,tournament_selection_size):
    selected = []
    for _ in range(len(population)-elite_size):
        tournament = random.sample(population,tournament_selection_size)
        winner = min(tournament, key=calculate_fitness)
        selected.append(winner)
    selected.extend(sorted(population,key=calculate_fitness)[:elite_size])
    return selected

def roulette_wheel_selection(population):
    ranked_population = sorted (population, key =calculate_fitness)
    fitness_sum = sum(1/ (rank+1)for rank in range (len(ranked_population)))
    selection_probabilities =[(1/(rank+1)) / fitness_sum for rank in range(len(ranked_population))]
    selected=random.choices(ranked_population,weights=selection_probabilities,k=len(population)-elite_size)
    selected.extend(sorted(population, key=calculate_fitness)[:elite_size])
    return selected


def boltzmann_selection(population, beta):
    ranked_population = sorted(population, key=calculate_fitness)
    fitness_sum = sum(math.exp(calculate_fitness(chromosome) * beta * -1) for chromosome in ranked_population)
    selection_probabilities = [math.exp(calculate_fitness(chromosome) * beta * -1) / fitness_sum for chromosome in ranked_population]
    selected = random.choices(ranked_population, weights=selection_probabilities, k=len(population)-elite_size)
    selected.extend(sorted(population, key=calculate_fitness)[:elite_size])
    return selected


#%% Hauptalgo
def genetic_algorithm(selection_method):
    population = generate_population(population_size) #Initialisiert Random 1. Loesung
    best_distances =[] #Definiert leere Liste fuer das plotten der Besten Distanz / Generation
    
    #Selektion entsprechd dem Auswahlverfahren
    for generation in range(num_generations):
        if selection_method == 'tournament':
            population = tournament_selection(population,tournament_selection_size)
        elif selection_method == 'roulette_wheel':
            population = roulette_wheel_selection(population)
        elif selection_method == 'boltzmann':
            population = boltzmann_selection(population, beta = 0.01)
        else:
            raise ValueError("UngÃ¼ltige Selektionsmethode. GÃ¼ltige Optionen sind 'Turnierauswahlverfahren', 'Roulette-Verfahren' und 'Stochastic Universal Sampling'.")              
        
        #Elitism
        next_generation = []
        next_generation.extend(sorted(population, key=calculate_fitness)[:elite_size])
        #Crossover and Mutation - ELITISM
        while len(next_generation)<population_size:
            parent1, parent2 = random.sample(population,2)
            child1=crossover(parent1,parent2)
            child2=crossover(parent2,parent1)
            next_generation.extend([mutate(child1),mutate(child2)])

        population = next_generation[:population_size]#Truncate, if population size is slightly larger (case if elitism=odd)
        #Best_chromosome for distance Plots
        best_chromosome = min(population, key=calculate_fitness) #--> JETZT EINEN INDENT MEHR!!! --> Da in jeder Iteration die Entfernung getrackt werden muss!
        best_distance = calculate_fitness(best_chromosome)
        best_distances.append(best_distance) #Fuegt die beste Distanz / Generation der Liste hinzu.
    
  # Rotate Matrix --> Auswahl Startstadt
    starting_index = best_chromosome.index(starting_city) #Dreht das Chromosome im folgnenden so, dass AusgewÃ¤hlte Startstadt am Anfang
    best_chromosome = best_chromosome[starting_index:] + best_chromosome[:starting_index] 
    best_chromosome.append(best_chromosome[0]) #fuegt die Startstadt wieder als Endstadt fuer Ausgabe hinzu, Distanz wurde bereits in Fitnessfunktion beruecksichtigt, ohne dass Eintrag in Liste vorhanden sein muss (da nach for Schliefe einmal distance chrom.[-1],chrom.[0])

  # Ausgabe:
    print(f'Selection Method: {selection_method}')
    print("Best Route:", "->".join(best_chromosome))
    print("Best Distance:", best_distance, "\n")
   
    data_x = list(range(num_generations))
    data_y = best_distances
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_x, y=data_y, mode='lines', name='Distanz'))
    fig.update_layout(
        width=800,
        height=600, 
        hovermode="x unified",
        xaxis = dict(showgrid = True,
                     gridcolor = 'lightgrey'),
        yaxis = dict(showgrid = True,
                    gridcolor = 'lightgrey',
                    range=[min(data_y)-10, max(data_y)+10] 
                    ),
        title=f'Selektionsverfahren: {selection_method}<br>Kürzeste Distanz: {best_distance}',
        xaxis_title='Generation',
        yaxis_title='Distanz der Route',
        plot_bgcolor='white',
        paper_bgcolor='#f2f2f2',
        annotations=[
            dict(
                x=1,
                y=1,
                text=f"Populationsgröße: {population_size}<br>Mutationsrate: {mutation_rate} <br>Generationen: {num_generations}",
                xref="paper",
                yref="paper",
                align= 'right'
            )
        ]

        )
    fig.show()
    # diagram_name = f'Mutate_Distanz_Konvergenz_{selection_method}'
    # # file_path = os.path.join('C:\\Users\\janne\\OneDrive\\Dokumente\\Seminar\\Kurv_Plots', f"{diagram_name}.html")
    # # pio.write_html(fig, file=file_path, auto_open=True)





    x = [city_coordinates[city][0] for city in best_chromosome]
    y = [city_coordinates[city][1] for city in best_chromosome]

    distances = []
    for i in range(len(best_chromosome) - 1):
        city1 = best_chromosome[i]
        city2 = best_chromosome[i + 1]
        distance = calculate_distance(city1, city2)
        distances.append(distance)
    print(distances)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Route', line=dict(color='black', width=3), showlegend=False))

    # distances_text = [f'Distanz {i+1}: {dist:.2f}' for i, dist in enumerate(distances)]
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers+text', name='Nodes',
                         marker=dict(color='lightblue', size=25, line=dict(color='black', width=2)),
                         textposition='top center', showlegend=False))

    
    fig.update_layout(
        hovermode="x unified",
        xaxis_title='X',
        yaxis_title='Y',
        title=f'Selektionsverfahren: {selection_method}<br>Distanz: {best_distance}',
        width=800,
        height=600, 
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor='white',
        paper_bgcolor='#f2f2f2',
        annotations=[
            dict(
                x=1,
                y=1,
                showarrow=False,
                text=f"Populationsgröße: {population_size}<br>Mutationsrate: {mutation_rate} <br>Generationen: {num_generations}",
                xref="paper",
                yref="paper",
                align= 'right'
            )
        ]
    )

    for i in range(len(best_chromosome)):
        fig.add_annotation(x=x[i], y=y[i], text=best_chromosome[i], showarrow=False, font=dict(size=16),
                        xshift=0, yshift=0, xanchor='center', yanchor='middle')

        fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode='markers', name='Starting City', marker=dict(color='red', size=30, line=dict(color='black', width=2)), showlegend=False))
        fig.add_annotation(x=x[0], y=y[0], text=best_chromosome[0], showarrow=False, font=dict(size=16),
                        xshift=0, yshift=0, xanchor='center', yanchor='middle')
    for i in range(len(best_chromosome) - 1):
        fig.add_annotation(
            x=x[i+1], y=y[i+1], ax=x[i], ay=y[i],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='black', standoff=10, startstandoff=10
    )
    fig.show()
    # diagram_name_2 = f'Mutate_Route_{selection_method}'
    # file_path_2 = os.path.join('C:\\Users\\janne\\OneDrive\\Dokumente\\Seminar\\Kurv_Plots', f"{diagram_name_2}.html")
    # pio.write_html(fig, file=file_path_2, auto_open=True)



#%% Ausführung

genetic_algorithm('tournament')
genetic_algorithm('roulette_wheel')
genetic_algorithm('boltzmann')