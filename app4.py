from math import e, sin, pi, pow
from random import randint


def fun(x):
    #  min +- 0.96 for x=2.35 where xe[0.5, 2.5]
    #  max +- 10.13 for x=2.45 where xe[0.5, 2.5]
    inside_sin = 10.0 * pi * x
    up = pow(e,x)*sin(inside_sin)+1
    bottom = x
    return up/bottom + 5.0


def decimal_to_binary_str(number):
    res = bin(number)
    res = str(res)
    res = list(res[2:])
    while len(res) != 18:
        res.insert(0,'0')
    result = ""
    return result.join(res)


def binary_str_to_decimal(binary_str):
    binary_str = str(binary_str)
    res = int(binary_str, 2)
    return res


def is_in_limit(value, v_min = 50000, v_max=250000):
    if v_min <= value <= v_max:
        return True
    else:
        return False


class GA:
    def __init__(self):
        self.population = []
        self.winners = []
        self.top_winners = []
        self.after_crossover = []
        self.after_mutation = []

    def init_population(self, population_size):
        while len(self.population) < population_size:
            x = randint(50000, 250000)
            if x not in self.population:
                self.population.append(x)
        for i in range(len(self.population)):
            self.population[i] = decimal_to_binary_str(self.population[i])

    #0:y -> move to next population
    #y:x -> move to winners and then there are used to mutation and crossover
    def select_x_best(self,y,x):
        # (value, index)
        winners = [(0,0)]
        for index,item in enumerate(self.population):
            value = fun(binary_str_to_decimal(item)/100000)
            for place, winner in enumerate(winners):
                if value > winner[0]:
                    winners.insert(place, (value, index))
                    break
        print(f"Current best value is {winners[0][0]} for x={binary_str_to_decimal(self.population[winners[0][1]])/100000}")
        res = [self.population[winner[1]] for winner in winners[y:x]]
        self.top_winners = []
        self.top_winners = [self.population[winner[1]] for winner in winners[:y]]
        self.winners = res

    def crossover(self, point):
        for i in range(len(self.winners)):
            for j in range(i+1,len(self.winners)):
                new_a = self.winners[i][:point] + self.winners[j][point:]
                new_b = self.winners[j][:point] + self.winners[i][point:]
                if is_in_limit(binary_str_to_decimal(new_a)):
                    self.after_crossover.append(new_a)
                if is_in_limit(binary_str_to_decimal(new_b)):
                    self.after_crossover.append(new_b)

    def mutation(self, gens_to_mutate):
        places = []
        while len(places)!= gens_to_mutate:
            x = randint(0,17)
            if x not in places:
                places.append(x)
        for winner in self.winners:
            mutated_winner = list(winner)
            for position in places:
                if mutated_winner[position] == '0':
                    mutated_winner[position] = '1'
                else:
                    mutated_winner[position] = '0'
            result = "".join(mutated_winner)
            if is_in_limit(binary_str_to_decimal(result)):
                self.after_mutation.append("".join(result))

    def clear(self):
        self.population = []

        for item in self.after_mutation:
            self.population.append(item)
        self.after_mutation = []

        for item in self.after_crossover:
            self.population.append(item)
        self.after_crossover = []

        for item in self.top_winners:
            self.population.append(item)
        self.top_winners = []

        self.winners = []

    def run(self, epochs, population_size):
        self.init_population(population_size)
        for epoch in range(epochs):
            self.select_x_best(2,6)
            self.crossover(9)
            self.mutation(3)
            self.clear()



ga = GA()
ga.run(1000, 10)


