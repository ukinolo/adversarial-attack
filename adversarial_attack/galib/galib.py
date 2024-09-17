from typing import Any, List
import adversarial_attack.galib.infrastructure as infrastructure

import numpy as np
import random
from enum import Enum

#Checking things
from typing import Callable
import warnings

import time

from tqdm import tqdm

def check_parameter(obj, param_name, method_name, strategy_name, default_value, valid_types, kwargs):
    '''
    Function creates property named params in object obj. params is dictionary that will contain value from kwargs with string key param_name.
    If value that is in kwargs does not meet the valid_types, value will be set to default_value.
    kwargs is also a dictionary
    '''
    try:
        obj.params
    except AttributeError: #obj.params does not exist, so we are going to create it
        obj.params = {}
    try:
        obj.params[param_name] = kwargs[param_name]
    except KeyError:
        obj.params[param_name] = None
    if(obj.params[param_name] == None):
        warnings.warn(f'Use {param_name}={valid_types} kayword argument when using {method_name} {strategy_name}')
    if(not isinstance(obj.params[param_name], valid_types)):
        obj.params[param_name] = None
        warnings.warn(f'{param_name} must be {valid_types} type')
    if(obj.params[param_name] is None):
        obj.params[param_name] = default_value
        warnings.warn(f'Setting default value for {param_name}={default_value}')

class Fitness():
    def _neighbor_diversity_control(self, population: np.array) -> np.array:
        threshold = self.params['threshold']
        penalty = np.zeros(shape=(population.shape[0], ))
        for i in range(len(population)):
            individual = population[i]
            for j in range(len(population)):
                other = population[j]
                distance = sum([abs(individual[k] - other[k]) for k in range(len(individual))])
                if distance < threshold:
                    penalty[i] += 1
        return penalty

    VALID_DIVERSITY_CONTROL_STRATS = ['neighbor',
                                     'None']
    DEFAULT_THRESHOLD = 0.1

    def get_possible_strats(self):
        return str(self.VALID_DIVERSITY_CONTROL_STRATS)

    def _check_strat(self, diversity_control_strat: str):
        if not diversity_control_strat in self.VALID_DIVERSITY_CONTROL_STRATS:
            raise ValueError('Diversity strategy name is not recognised, you can use diversity_control_strat=' + self.get_possible_strats())

    def _set_diversity_control(self, diversity_control_strat: str, **kwargs):
        if(diversity_control_strat == 'neighbor'):
            self._diversity_control = self._neighbor_diversity_control
            check_parameter(self, 'threshold', 'neighbor', 'diversity control', self.DEFAULT_THRESHOLD, (int, float), kwargs)
            self.params['threshold'] = float(self.params['threshold'])

        elif(diversity_control_strat == 'None'):
            self._diversity_control = None
        
        else:
            raise AssertionError('Fitness diversity control name exist in the valid diversity control strategies but was not found')
    
    def __init__(self, fitness_function: Callable, system_size, *, diversity_control_strat: str = 'None', threshold: float | None = None):
        if(not isinstance(fitness_function, Callable)):
            raise TypeError('fitness_function must be Callable')
        try:
            individual = np.zeros(system_size)
            fitness_function(individual)
        except IndexError:
            raise IndexError('For your fitness function, this system size is too small')
        self.system_size = system_size
        self._diversity_control_strat = diversity_control_strat
        self._check_strat(diversity_control_strat)
        self._set_diversity_control(diversity_control_strat, threshold=threshold)
        self.function = fitness_function

    def get_info(self):
        return f'Fitness have diversity control - {self._diversity_control_strat} with next parameters:'+ ('\n\t'.join("{0} = {1}".format(key, value)  for key,value in self.params.items()) if self.params.items() else f'\n\tMethod have no parameters') 

    def calculate_fitness(self, population: np.array) -> np.array:
        fitness_values = np.apply_along_axis(self.function, 1, population)
        penalty = self._diversity_control(population) if self._diversity_control != None else None
        return fitness_values/penalty if penalty != None else fitness_values

    def calculate_simple_fitness(self, population: np.array) -> np.array:
        return np.apply_along_axis(self.function, 1, population)

    def funciton(self, individual: np.array) -> float:
        return self.function(individual)

    def rank(self, population: np.array) -> np.array:
        population_fitness = self.calculate_fitness(population)
        return population[np.argsort(population_fitness)[::-1]]



class Selection():
    def _roulette_wheel_selection(self, population: np.array, fitness: Fitness, number_of_individuals_to_select: int):
        population_fitness = fitness.calculate_fitness(population)
        total_fitness = sum(population_fitness)
            
        weights = ()
        for fitness_value in population_fitness:
            if(fitness_value < 0):
                raise ValueError('Error fitness value can\'t be negative')
            weights += (fitness_value/total_fitness, )
        
        return np.array(random.choices(population, weights, k=number_of_individuals_to_select))
    _roulette_wheel_method = infrastructure.Method('roulette wheel',
                                    _roulette_wheel_selection,
                                    parameters=[])


    def _scaled_roulette_wheel_selection(self, population: np.array, fitness: Callable, number_of_individuals_to_select: int):
        population_fitness = fitness.calculate_fitness(population)
        min_fitness = min(population_fitness)
        if min_fitness < 0:
            raise ValueError('Error fitness value can\'t be negative')
            
        weights = ()
        for fitness_value in population_fitness:
            weights += (fitness_value/min_fitness, )
        
        return np.array(random.choices(population, weights, k=number_of_individuals_to_select))
    _scaled_roulette_wheel_method = infrastructure.Method('scaled roulette wheel',
                                           _scaled_roulette_wheel_selection,
                                           parameters=[])


    def _ranking_selection(self, population: np.array, fitness: Fitness, number_of_individuals_to_select: int):
        max_rank_multiplier = self['max_rank_multiplier']
        population = fitness.rank(population)
        weights = ()
        for i in range(len(population)):
            weights += (round((1 - max_rank_multiplier)/(len(population) - 1) * i + max_rank_multiplier, 3), )
        
        return np.array(random.choices(population, weights, k=number_of_individuals_to_select))
    _ranking_method = infrastructure.Method('ranking',
                             _ranking_selection,
                             parameters=[
                                 infrastructure.Parameter('max_rank_multiplier', (float, ), 20.0)
                             ])


    def _tournament_selection(self, population: np.array, fitness: Fitness, number_of_individuals_to_select: int):
        tournament_size = self['tournament_size']
        mating_pool = np.empty(shape=(number_of_individuals_to_select, population.shape[1]))
        population_fitness = fitness.calculate_fitness(population)
        population_size = len(population)
        for i in range(number_of_individuals_to_select):
            tournament_indexes = [random.randrange(population_size) for _ in range(tournament_size)]
            tournament_fitnesses = population_fitness[tournament_indexes]
            winner_index = max(enumerate(tournament_fitnesses), key=lambda x: x[1])[0]
            mating_pool[i] = population[tournament_indexes[winner_index]]
        return mating_pool
    _tournament_method = infrastructure.Method('tournament',
                                _tournament_selection,
                                parameters=[
                                    infrastructure.Parameter('tournament_size', (int, ), 3)
                                ])


    def _random_selection(self, population: np.array, fitness: Fitness, number_of_individuals_to_select: int):
        return np.array(random.choices(population, k=number_of_individuals_to_select))
    _random_method = infrastructure.Method('random',
                            _random_selection,
                            parameters=[])

    _info = infrastructure.Strategy('Selection')
    _info.add_method(_roulette_wheel_method)
    _info.add_method(_scaled_roulette_wheel_method)
    _info.add_method(_ranking_method)
    _info.add_method(_tournament_method)
    _info.add_method(_random_method)

    def __init__(self, method_name: str, *, max_rank_multiplier: Any=None, tournament_size: Any=None) -> None:
        arguments = locals()
        arguments.pop('self')
        arguments.pop('method_name')
        arguments_to_set = {}
        assert isinstance(method_name, str)
        self._method = Selection._info.get_method_copy(method_name)
        for parameter_name in self._method.get_parameter_names():
            if arguments[parameter_name] == None:
                warnings.warn(f'GALIB warning: you can use {parameter_name} keyword argument if you do not want to use default value')
            else:
                arguments_to_set[parameter_name] = arguments[parameter_name]
        self._method.set_parameter_values(**arguments_to_set)

    def __call__(self, population: np.array, fitness: Fitness, number_of_individuals_to_select: int) -> np.array:
        return self._method(population=population, fitness=fitness, number_of_individuals_to_select=number_of_individuals_to_select)
    
    def get_info(self) -> str:
        return 'Selection is using ' + self._method.get_info()
    
    def get_description(self) -> str:
        return Selection._info.get_description(number_of_tabs=0)



class Crossover():
    def _symmetrical_crossover(self, mating_pool: np.array, fitness: Fitness):
        sistem_size = len(mating_pool[0])
        offspring = np.empty(shape=mating_pool.shape)
        
        for i in range(0, len(mating_pool), 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i+1]
            child1 = np.empty(sistem_size)
            child2 = np.empty(sistem_size)
            omega = random.random()
            for j in range(sistem_size):
                child1[j] = parent1[j] * omega + parent2[j] * (1 - omega)
                child2[j] = parent1[j] * (1 - omega) + parent2[j] * omega
            offspring[i] = child1
            offspring[i+1] = child2
        return offspring
    _symmetrical_method = infrastructure.Method('symmetrical',
                                 _symmetrical_crossover,
                                 parameters=[])

    def _linear_crossover(self, mating_pool: np.array, fitness: Fitness):
        sistem_size = len(mating_pool[0])
        offspring = np.empty(shape=mating_pool.shape)

        for i in range(0, len(mating_pool), 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i+1]
            child1 = np.empty(sistem_size)
            child2 = np.empty(sistem_size)
            child3 = np.empty(sistem_size)
            for j in range(sistem_size):
                child1[j] =  0.5 * parent1[j] + 0.5 * parent2[j]
                child2[j] = -0.5 * parent1[j] + 1.5 * parent2[j]
                child3[j] =  1.5 * parent1[j] - 0.5 * parent2[j]
                
            offspring[i:i+2] = sorted([child1, child2, child3], key=fitness.function, reverse=True)[0:2]

        return offspring
    _linear_method = infrastructure.Method('linear',
                            _linear_crossover,
                            parameters=[])

    def _bland_crossover(self, mating_pool: np.array, fitness: Fitness):
        sistem_size = len(mating_pool[0])
        offspring = np.empty(shape=mating_pool.shape)
        alpha = self['alpha']
        
        for i in range(0, len(mating_pool), 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i+1]
            child1 = np.empty(sistem_size)
            child2 = np.empty(sistem_size)
            for j in range(sistem_size):
                [lower, higher] = sorted([parent1[j], parent2[j]])
                distance = higher - lower
                child1[j] = random.random() * distance * (1 + 2 * alpha) + lower - (alpha * distance)#Scaling random.random() from [0,1] to desired interval
                child2[j] = random.random() * distance * (1 + 2 * alpha) + lower - (alpha * distance)
                
            offspring[i:i+2] = [child1, child2]

        return offspring
    _bland_method = infrastructure.Method('bland',
                           _bland_crossover,
                           parameters=[
                               infrastructure.Parameter('alpha', (float, ), 0.5)
                           ])

    def _sbx(self, mating_pool: np.array, fitness: Fitness):
        system_size = len(mating_pool[0])
        offspring = np.empty(shape=mating_pool.shape)
        distribution_index = self['distribution_index']
        probability_of_crossover = self['probability_of_crossover'] if system_size > 1 else 1
        
        for i in range(0, len(mating_pool), 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i+1]
            child1 = np.empty(system_size)
            child2 = np.empty(system_size)
            for j in range(system_size):
                u = random.random()
                if(u < 0.5):
                    u = 2 * u
                else:
                    u = 1.0/(2.0 * (1-u))
                beta = u**(1.0/distribution_index)
                if(random.random() < probability_of_crossover):
                    child1[j] = 0.5*((1+beta)*parent1[j] + (1-beta)*parent2[j])
                    child2[j] = 0.5*((1-beta)*parent1[j] + (1+beta)*parent2[j])
                else:
                    child1[j] = parent1[j]
                    child2[j] = parent2[j]

            offspring[i:i+2] = [child1, child2]

        return offspring
    _sbx_method = infrastructure.Method('sbx',
                         _sbx,
                         parameters=[
                             infrastructure.Parameter('distribution_index', (float, ), 1.5),
                             infrastructure.Parameter('probability_of_crossover', (float, ), 0.5)
                         ])
    
    _info = infrastructure.Strategy('Crossover')
    _info.add_method(_symmetrical_method)
    _info.add_method(_linear_method)
    _info.add_method(_bland_method)
    _info.add_method(_sbx_method)
    
    def __init__(self, method_name: str, *, alpha=None, distribution_index=None, probability_of_crossover=None):
        arguments = locals()
        arguments.pop('self')
        arguments.pop('method_name')
        arguments_to_set = {}
        assert isinstance(method_name, str)
        self._method = Crossover._info.get_method_copy(method_name)
        for parameter_name in self._method.get_parameter_names():
            if arguments[parameter_name] == None:
                warnings.warn(f'GALIB warning: you can use {parameter_name} keyword argument if you do not want to use default value')
            else:
                arguments_to_set[parameter_name] = arguments[parameter_name]
        self._method.set_parameter_values(**arguments_to_set)

    def __call__(self, mating_pool: np.array, fitness: Fitness) -> np.array:
        return self._method(mating_pool=mating_pool, fitness=fitness)
        
    def get_info(self) -> str:
        return 'Crossover is using ' + self._method.get_info()
    
    def get_description(self) -> str:
        return Crossover._info.get_description(number_of_tabs=0)

class Replacement():
    def _best_n_replacement(self, offspring: np.array, offspring_fitness: np.array, population: np.array, population_fitness: np.array) -> np.array:
        everyone = np.concatenate((offspring, population))
        everyone_fitness = np.concatenate((offspring_fitness, population_fitness))
        sorted_all = np.flip(everyone[np.argsort(everyone_fitness)], axis=0)
        return sorted_all[:len(population)]
    _best_n_method = infrastructure.Method('best',
                            _best_n_replacement,
                            parameters=[])


    def _elitism_replacement(self, offspring: np.array, offspring_fitness: np.array, population: np.array, population_fitness: np.array) -> np.array:
        elitism_width = self['elitism_width']
        sorted_offspring = np.flip(offspring[np.argsort(offspring_fitness)], axis=0)
        sorted_population = np.flip(population[np.argsort(population_fitness)], axis=0)
        number_survived_parents = round(len(population) * elitism_width)
        number_offspring_survived = len(population) - number_survived_parents
        return np.concatenate((sorted_population[:number_survived_parents], sorted_offspring[:number_offspring_survived]))
    _elitism_method = infrastructure.Method('elitism',
                             _elitism_replacement,
                             parameters=[
                                 infrastructure.Parameter('elitism_width', (float, ), 0.05)
                             ])


    def _random_replacement(self, offspring: np.array, offspring_fitness: np.array, population: np.array, population_fitness: np.array) -> np.array:
        everyone = np.concatenate((offspring, population))
        return np.array(random.choices(everyone, k=len(population)))
    _random_method = infrastructure.Method('random',
                            _random_replacement,
                            parameters=[])

    _info = infrastructure.Strategy('Replacement')
    _info.add_method(_best_n_method)
    _info.add_method(_elitism_method)
    _info.add_method(_random_method)

    def __init__(self, method_name: str, *, elitism_width=None):
        arguments = locals()
        arguments.pop('self')
        arguments.pop('method_name')
        arguments_to_set = {}
        assert isinstance(method_name, str)
        self._method = Replacement._info.get_method_copy(method_name)
        for parameter_name in self._method.get_parameter_names():
            if arguments[parameter_name] == None:
                warnings.warn(f'GALIB warning: you can use {parameter_name} keyword argument if you do not want to use default value')
            else:
                arguments_to_set[parameter_name] = arguments[parameter_name]
        self._method.set_parameter_values(**arguments_to_set)

    def __call__(self, offspring: np.array, offspring_fitness: np.array, population: np.array, population_fitness: np.array) -> np.array:
        return self._method(offspring=offspring, offspring_fitness=offspring_fitness,
                            population=population, population_fitness=population_fitness)

    def get_info(self) -> str:
        return 'Replacement is using ' + self._method.get_info()
    
    def get_description(self) -> str:
        return Replacement._info.get_description(number_of_tabs=0)




class Mutation():
    def _random_mutation(self, population: np.array, mutation_probability: float, dimension_probability: float):
        mutation_strength = self['mutation_strength']
        dimension_probability = dimension_probability if len(population[0]) > 1 else 1 #If there is more than 1 dimension we look at dimension_probability
        for i in range(len(population)):
            if(random.random() > mutation_probability): continue
            individual = population[i]
            for j in range(len(individual)):
                if(random.random() > dimension_probability): continue
                population[i][j] += mutation_strength * (2 * (random.random() - 0.5))
    _random_method = infrastructure.Method('random',
                            _random_mutation,
                            parameters=[
                                infrastructure.Parameter('mutation_strength', (float, ), 0.2)
                            ])

    #Non uniform mutation is not currently working because of change in infrastructure
    def _nonuniform_mutation(self, population: np.array, mutation_probability: float, dimension_probability: float):
        maximum_generations = self._maximum_generations
        current_generation = self._current_generation
        mutation_strength = self['mutation_strength']
        b = self['b']
        dimension_probability = dimension_probability if len(population[0]) > 1 else 1
        
        for i in range(len(population)):
            if(random.random() > mutation_probability): continue
            individual = population[i]
            for j in range(len(individual)):
                if(random.random() > dimension_probability): continue
                tau = -1 if random.randrange(2) == 0 else 1
                population[i,j] += tau*(mutation_strength)*(1 - random.random()**(1-current_generation/maximum_generations)**b)
    _nonuniform_method = infrastructure.Method('nonuniform',
                                _nonuniform_mutation,
                                parameters=[
                                    infrastructure.Parameter('mutation_strength', (float, ), 0.2),
                                    infrastructure.Parameter('b', (float, ), 1.0)
                                ])

    def _normally_distributed_mutation(self, population: np.array, mutation_probability: float, dimension_probability: float):
        standard_deviation = self['standard_deviation']
        for i in range(len(population)):
            if(random.random() > mutation_probability): continue
            individual = population[i]
            for j in range(len(individual)):
                if(random.random() > dimension_probability): continue
                population[i,j] += random.gauss(mu=0, sigma=standard_deviation)
    _normally_distributed_method = infrastructure.Method('normal',
                                          _normally_distributed_mutation,
                                          parameters=[
                                              infrastructure.Parameter('standard_deviation', (float, ), 1.0)
                                          ])

    _info = infrastructure.Strategy('Mutation')
    _info.add_method(_random_method)
    _info.add_method(_nonuniform_method)
    _info.add_method(_normally_distributed_method)

    def __init__(self, method_name: str, mutation_probability: float, dimension_probability: float, *, mutation_strength=None, b=None, standard_deviation=None):
        arguments = locals()
        arguments.pop('self')
        arguments.pop('method_name')
        arguments_to_set = {}
        assert isinstance(method_name, str)
        self._method = Mutation._info.get_method_copy(method_name)
        for parameter_name in self._method.get_parameter_names():
            if arguments[parameter_name] == None:
                warnings.warn(f'GALIB warning: you can use {parameter_name} keyword argument if you do not want to use default value')
            else:
                arguments_to_set[parameter_name] = arguments[parameter_name]
        self._method.set_parameter_values(**arguments_to_set)
        self._mutation_probability = mutation_probability
        self._dimension_probability = dimension_probability

    def __call__(self, population: np.array) -> np.array:
        return self._method(population=population, mutation_probability=self._mutation_probability, dimension_probability=self._dimension_probability)

    def get_info(self) -> str:
        return f'Mutation have mutation probability = {self._mutation_probability} ' +\
                f' and dimension probability = {self._dimension_probability} and is using\n' +\
                    self._method.get_info()
    
    def get_description(self) -> str:
        return Mutation._info.get_description(number_of_tabs=0)

    def initialize(self, maximum_generations: int):
        self._maximum_generations = maximum_generations
        self._current_generation = 1

    def update(self):
        self._current_generation += 1


class GA():
    def __init__(self, fitness: Fitness | List[Fitness], selection: Selection, crossover: Crossover, replacement: Replacement, mutation: Mutation, change_fitness_function: Callable = None):
        self._population = None
        self._selection = selection
        self._crossover = crossover
        self._replacement = replacement
        self._mutation = mutation
        if change_fitness_function == None:
            self._fitness = fitness
            self._fitnesses = None
            self._change_fitness_function = None
        else:
            self._fitness = fitness[0]
            self._fitnesses = fitness
            self._change_fitness_function = change_fitness_function

    def initialize_population_interval(self, number_of_individuals: int, interval_min: int = 0, interval_max: int = 0):
        system_size = self._fitness.system_size
        self._population = np.empty((number_of_individuals, system_size))
        for i in range(number_of_individuals):
            individual = np.empty((system_size, ))
            for j in range(system_size):
                individual[j] = random.uniform(interval_min, interval_max)
            self._population[i] = np.copy(individual)
        self._population = self._fitness.rank(self._population)

    def initialize_population_value(self, number_of_individuals: int, value: Any, mutation_probability: float = 0, dimension_probability: float = 0, mutation_strength: float = 0):
        system_size = self._fitness.system_size
        assert(system_size == len(value))
        self._population = np.full((number_of_individuals,system_size), value, dtype=object)
        if(mutation_probability > 0):
            Mutation("random", mutation_probability, dimension_probability, mutation_strength=mutation_strength).__call__(self._population)
        self._population = self._fitness.rank(self._population)

    # def initialize_population(self, number_of_individuals: int, generate_individual: Callable):
    #     system_size = self._fitness.system_size
    #     self._population = np.zeros((number_of_individuals,system_size))
    #     for i in range(len(self._population)):
    #         self._population[i] = generate_individual()
    #     self._population = self._fitness.rank(self._population)

    def run_w_i(self, number_of_individuals: int, rounds, *, save=False, file_name='ga', verbose=0, early_stopping_rounds=-1, run_times=1, fitness_function_name='Not given'):
        #Variables for storing all runs best individuals and other info
        best_individuals = []
        best_fitness_values = []
        total_rounds = []
        times = []
        fitness_history = []
        
        for k in range(run_times):
            start_time = time.time()
            single_run_fitness_history = []

            max_ind = self._population[0]
            max_fit = self._fitness.funciton(max_ind)
            
            self._mutation.initialize(rounds)
            round_without_new_max = 0
            #pbar = tqdm(range(rounds))
            for i in range(rounds):
                round_without_new_max += 1
                
                mating_pool = self._selection(self._population, self._fitness, number_of_individuals)
                offspring = self._crossover(mating_pool, self._fitness)
                self._mutation(offspring)
                self._population = self._replacement(offspring, self._fitness.calculate_fitness(offspring),
                                                     self._population, self._fitness.calculate_fitness(self._population))
                self._mutation.update()
                self._population = self._fitness.rank(self._population)
                round_max_fit = self._fitness.funciton(self._population[0])
                single_run_fitness_history.append(round_max_fit)

                #Printing progress
                if verbose and verbose > 0:
                    if i % verbose == 0:
                        print(f'Printing info for run {k} and round {i}:\n'
                             +f'\t Current max fit:{max_fit}\n'
                             +f'\t Current max ind:{max_ind}\n'
                             +f'\t Round max fit:{round_max_fit}\n'
                             +f'\t Round max ind:{self._population[0]}\n')

                #Checking for new best
                if(round_max_fit > max_fit):
                    max_fit = round_max_fit
                    max_ind = self._population[0]
                    round_without_new_max = 0

                #Checking for early stropping
                if(round_without_new_max >= early_stopping_rounds and early_stopping_rounds > 0):
                    break
            end_time = time.time()

            #Storing run info
            best_individuals.append(max_ind)
            best_fitness_values.append(max_fit)
            total_rounds.append(i)
            times.append(end_time-start_time)
            fitness_history.append(single_run_fitness_history)
        #End of all runs

        return best_individuals, best_fitness_values, total_rounds, times, fitness_history

    def run_mf(self, number_of_individuals: int, rounds, *, save=False, file_name='ga', verbose=0, early_stopping_rounds=-1, run_times=1, fitness_function_name='Not given'):
        #Variables for storing all runs best individuals and other info
        best_individuals = []
        best_fitness_values = []
        total_rounds = []
        times = []
        fitness_history = []
        
        for k in range(run_times):
            start_time = time.time()
            single_run_fitness_history = []
            self._fitness = self._fitnesses[0]

            max_ind = self._population[0]
            max_fit = [-np.inf] * len(self._fitnesses)
            max_fit[0] = self._fitness.funciton(max_ind)
            fitness_function_index = 0
            
            self._mutation.initialize(rounds)
            round_without_new_max = 0
            #pbar = tqdm(range(rounds))
            for i in range(rounds):
                round_without_new_max += 1
                
                mating_pool = self._selection(self._population, self._fitness, number_of_individuals)
                offspring = self._crossover(mating_pool, self._fitness)
                self._mutation(offspring)
                self._population = self._replacement(offspring, self._fitness.calculate_fitness(offspring),
                                                     self._population, self._fitness.calculate_fitness(self._population))
                self._mutation.update()
                self._population = self._fitness.rank(self._population)
                round_max_fit = self._fitness.funciton(self._population[0])
                single_run_fitness_history.append(round_max_fit)

                #Printing progress
                if verbose and verbose > 0:
                    if i % verbose == 0:
                        print(f'Printing info for run {k} and round {i}:\n'
                             +f'\t Current max fit:{max_fit}\n'
                             +f'\t Current max ind:{max_ind}\n'
                             +f'\t Round max fit:{round_max_fit}\n'
                             +f'\t Round max ind:{self._population[0]}\n')

                #Checking for new best
                if(round_max_fit > max_fit[fitness_function_index]):
                    max_fit[fitness_function_index] = round_max_fit
                    max_ind = self._population[0]
                    round_without_new_max = 0

                #Checking for early stropping
                if(round_without_new_max >= early_stopping_rounds and early_stopping_rounds > 0):
                    break
                fitness_function_index = self._change_fitness_function(max_ind, k)
                self._fitness = self._fitnesses[fitness_function_index]
            end_time = time.time()

            #Storing run info
            best_individuals.append(max_ind)
            best_fitness_values.append(max_fit)
            total_rounds.append(i)
            times.append(end_time-start_time)
            fitness_history.append(single_run_fitness_history)
        #End of all runs

        return best_individuals, best_fitness_values, total_rounds, times, fitness_history

    def run(self, number_of_individuals: int, interval_min: int, interval_max: int, rounds, *, save=False, file_name='ga', verbose=0, early_stopping_rounds=-1, run_times=1, fitness_function_name='Not given'):
        
        #Variables for storing all runs best individuals and other info
        best_individuals = []
        best_fitness_values = []
        total_rounds = []
        times = []
        fitness_history = []
        
        for k in range(run_times):
            start_time = time.time()
            single_run_fitness_history = []

            self.initialize_population(number_of_individuals, interval_min, interval_max)
            max_ind = self._population[0]
            max_fit = self._fitness.funciton(max_ind)
            
            self._mutation.initialize(rounds)
            round_without_new_max = 0
            #pbar = tqdm(range(rounds))
            for i in range(rounds):
                round_without_new_max += 1
                
                mating_pool = self._selection(self._population, self._fitness, number_of_individuals)
                offspring = self._crossover(mating_pool, self._fitness)
                self._mutation(offspring)
                self._population = self._replacement(offspring, self._fitness.calculate_fitness(offspring),
                                                     self._population, self._fitness.calculate_fitness(self._population))
                self._mutation.update()
                self._population = self._fitness.rank(self._population)
                round_max_fit = self._fitness.funciton(self._population[0])
                single_run_fitness_history.append(round_max_fit)

                #Printing progress
                if verbose and verbose > 0:
                    if i % verbose == 0:
                        print(f'Printing info for run {k} and round {i}:\n'
                             +f'\t Current max fit:{max_fit}\n'
                             +f'\t Current max ind:{max_ind}\n'
                             +f'\t Round max fit:{round_max_fit}\n'
                             +f'\t Round max ind:{self._population[0]}\n')

                #Checking for new best
                if(round_max_fit > max_fit):
                    max_fit = round_max_fit
                    max_ind = self._population[0]
                    round_without_new_max = 0

                #Checking for early stropping
                if(round_without_new_max >= early_stopping_rounds and early_stopping_rounds > 0):
                    break
            end_time = time.time()
            
            #print(f'Time to execute {k} run is: {end_time - start_time}')
            #if(round_without_new_max >= early_stopping_rounds and early_stopping_rounds > 0):
                #print('Stopped due to reaching early_stopping_rounds')
            #print(f'Found solution in {i} round.')
            #print(f'Best individual fitness: {max_fit}')
            #print(f'Individual value is: {max_ind}')

            #Storing run info
            best_individuals.append(max_ind)
            best_fitness_values.append(max_fit)
            total_rounds.append(i)
            times.append(end_time-start_time)
            fitness_history.append(single_run_fitness_history)
        #End of all runs

        return best_individuals, best_fitness_values, total_rounds, times, fitness_history
    
    def get_info(self):
        return self._selection.get_info() + '\n' + self._crossover.get_info() + '\n' \
            + self._mutation.get_info() + '\n' + self._replacement.get_info()
    
    # def save_to_file(self, file_name: str):
    #     if not file_name.endswith('.txt'):
    #         file_name.removesuffix('.txt')
    #     try:
    #         f = open(file_name + '.txt', 'x')
    #     except FileExistsError:
    #         i = 0
    #         while True:
    #             try:
    #                 f = open(file_name+'_'+str(i)+'.txt', 'x')
    #                 break
    #             except FileExistsError:
    #                 i += 1
    #     #Have f file
    #     #TODO write
    #     f.write(self._selection.get_info())
    #     f.write(f'\n')
    #     f.write(self._crossover.get_info())
    #     f.write(f'\n')
    #     f.write(self._mutation.get_info())
    #     f.write(f'\n')
    #     f.write(self._replacement.get_info())

    # def read_from_file(self):
    #     #TODO
    #     pass


