#!/usr/bin/env python3
import sys
import re
import random
import numpy as np


# - accept a single input file
# - parse it for the clauses
# - solve it recursivey with David Putnam 


class sat_solvers:

    def __init__(self):
        self.clauses = []
        self.variables = []
        self.literals = []
        self.truth_values = dict()
        # self.truth_clauses = dict()

    def parse(self, filename):
        statLine = re.compile('p\s*cnf\s*(\d*)\s*(\d*)') #Regular expression to detect stat line

        rules_dimacs = open(filename, 'r')
        rules_lines = rules_dimacs.read().split('\n')
        for line in rules_lines:
            line=line.strip()
            if line=="%" or not line:
                continue
            else:
                stats=statLine.match(line)
                if stats:
                    self.varCount=int(stats.group(1)) #number of unknowns
                    self.termCount=int(stats.group(2)) #number of clauses/terms
                else:
                    numbers=line.split()
                    literals = []
                    for number in numbers:
                        n=int(number)
                        # n=number
                        if(n!=0):
                            literals.append(n)
                            if n not in self.literals: self.literals.append(n)
                            if abs(n) not in self.variables: self.variables.append(abs(n))
                    self.clauses.append(literals)
        algo.truth_values = {var:False for var in algo.variables} 
        return


    def print_sudoku(self, true_vars):
        """
        Print sudoku.
        :param true_vars: List of variables that your system assigned as true. Each var should be in the form of integers.
        :return:
        """
        if len(true_vars) != 81:
            print("Wrong number of variables.")
            return
        s = []
        row = []
        for i in range(len(true_vars)):
            row.append(str(int(true_vars[i]) % 10))
            if (i+1) % 9 == 0:
                s.append(row)
                row = []

        print("╔═" + "═" + "═╦═" + "═" + "═╦═" + "═" + "═╦═" + "═" + "═╦═" + "═" + "═╦═" + "═" + "═╦═" + "═" + "═╦═" + "═" + "═╦═" + "═" + "═╗")
        print("║ "+s[0][0]+" | "+s[0][1]+" | "+s[0][2]+" ║ "+s[0][3]+" | "+s[0][4]+" | "+s[0][5]+" ║ "+s[0][6]+" | "+s[0][7]+" | "+s[0][8]+" ║")
        print("╠─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╬─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╬─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╣")
        print("║ "+s[1][0]+" | "+s[1][1]+" | "+s[1][2]+" ║ "+s[1][3]+" | "+s[1][4]+" | "+s[1][5]+" ║ "+s[1][6]+" | "+s[1][7]+" | "+s[1][8]+" ║")
        print("╠─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╬─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╬─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╣")
        print("║ "+s[2][0]+" | "+s[2][1]+" | "+s[2][2]+" ║ "+s[2][3]+" | "+s[2][4]+" | "+s[2][5]+" ║ "+s[2][6]+" | "+s[2][7]+" | "+s[2][8]+" ║")
        print("╠═" + "═" + "═╬═" + "═" + "═╬═" + "═" + "═╬═" + "═" + "═╬═" + "═" + "═╬═" + "═" + "═╬═" + "═" + "═╬═" + "═" + "═╬═" + "═" + "═╣")
        print("║ "+s[3][0]+" | "+s[3][1]+" | "+s[3][2]+" ║ "+s[3][3]+" | "+s[3][4]+" | "+s[3][5]+" ║ "+s[3][6]+" | "+s[3][7]+" | "+s[3][8]+" ║")
        print("╠─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╬─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╬─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╣")
        print("║ "+s[4][0]+" | "+s[4][1]+" | "+s[4][2]+" ║ "+s[4][3]+" | "+s[4][4]+" | "+s[4][5]+" ║ "+s[4][6]+" | "+s[4][7]+" | "+s[4][8]+" ║")
        print("╠─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╬─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╬─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╣")
        print("║ "+s[5][0]+" | "+s[5][1]+" | "+s[5][2]+" ║ "+s[5][3]+" | "+s[5][4]+" | "+s[5][5]+" ║ "+s[5][6]+" | "+s[5][7]+" | "+s[5][8]+" ║")
        print("╠═" + "═" + "═╬═" + "═" + "═╬═" + "═" + "═╬═" + "═" + "═╬═" + "═" + "═╬═" + "═" + "═╬═" + "═" + "═╬═" + "═" + "═╬═" + "═" + "═╣")
        print("║ "+s[6][0]+" | "+s[6][1]+" | "+s[6][2]+" ║ "+s[6][3]+" | "+s[6][4]+" | "+s[6][5]+" ║ "+s[6][6]+" | "+s[6][7]+" | "+s[6][8]+" ║")
        print("╠─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╬─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╬─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╣")
        print("║ "+s[7][0]+" | "+s[7][1]+" | "+s[7][2]+" ║ "+s[7][3]+" | "+s[7][4]+" | "+s[7][5]+" ║ "+s[7][6]+" | "+s[7][7]+" | "+s[7][8]+" ║")
        print("╠─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╬─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╬─" + "─" + "─┼─" + "─" + "─┼─" + "─" + "─╣")
        print("║ "+s[8][0]+" | "+s[8][1]+" | "+s[8][2]+" ║ "+s[8][3]+" | "+s[8][4]+" | "+s[8][5]+" ║ "+s[8][6]+" | "+s[8][7]+" | "+s[8][8]+" ║")
        print("╚═" + "═" + "═╩═" + "═" + "═╩═" + "═" + "═╩═" + "═" + "═╩═" + "═" + "═╩═" + "═" + "═╩═" + "═" + "═╩═" + "═" + "═╩═" + "═" + "═╝")

    def check_sudoku(self, true_vars):
        """
        Check sudoku.
        :param true_vars: List of variables that your system assigned as true. Each var should be in the form of integers.
        :return:
        """
        import math as m
        s = []
        row = []
        for i in range(len(true_vars)):
            row.append(str(int(true_vars[i]) % 10))
            if (i + 1) % 9 == 0:
                s.append(row)
                row = []

        correct = True
        for i in range(len(s)):
            for j in range(len(s[0])):
                for x in range(len(s)):
                    if i != x and s[i][j] == s[x][j]:
                        correct = False
                        print("Repeated value in column:", j)
                for y in range(len(s[0])):
                    if j != y and s[i][j] == s[i][y]:
                        correct = False
                        print("Repeated value in row:", i)
                top_left_x = int(i-i%m.sqrt(len(s)))
                top_left_y = int(j-j%m.sqrt(len(s)))
                for x in range(top_left_x, top_left_x + int(m.sqrt(len(s)))):
                    for y in range(top_left_y, top_left_y + int(m.sqrt(len(s)))):
                        if i != x and j != y and s[i][j] == s[x][y]:
                            correct = False
                            print("Repeated value in cell:", (top_left_x, top_left_y))
        return correct

class dpll(sat_solvers):

    def __init__(self):
        super().__init__()

    def bcp(self,formula_set, unit):
        modified = []
        for clause in formula_set:
            if unit in clause: continue
            if -unit in clause:
                c = [x for x in clause if x != -unit]
                # if len(c) == 0: return False
                modified.append(c)
            else:
                modified.append(clause)
        return modified


    def satisfiable(self, clause_set):
        unit_clauses = [c for c in clause_set if len(c) == 1]
        while len(unit_clauses) > 0:
            #take the literal of first clause
            lit = unit_clauses[0][0]
            if lit > 0: self.truth_values[lit] = True #setting truth value
            elif lit < 0: self.truth_values[abs(lit)] = False
            # unit propagation:
            clause_set = self.bcp(clause_set,lit)
            # check if the clause set is empty: (SAT)
            if len(clause_set)==0:
                return True
            # check if there is an empty clause: (UNSAT)
            for clause in clause_set:
                if len(clause)==0:
                    return False
            unit_clauses = [c for c in clause_set if len(c) == 1]
        #splitting:
        rand_lit = random.choice(random.choice(clause_set))
        
        if self.satisfiable(clause_set+[[rand_lit]]):
            return True
        elif self.satisfiable(clause_set+[[-rand_lit]]):
            return True
        else:
            return False

    def run(self):
        return self.satisfiable(self.clauses)

class aco(sat_solvers):

    def __init__(self):
        super().__init__()
        self.NUM_ANTS = 35
        # exponential factor for pheromones in probabilities, range: (-inf, inf)
        self.EXP_PH = 0.9
        # exponential factor for most constrained variable heuristic in probabilities, range: (-inf, inf)
        self.EXP_MCV = 0.1
        # pheromone reduce factor (per iteration), range: (0, 1)
        self.PH_REDUCE_FACTOR = 0.5
        # blur pheromones interval, range: [1, inf)
        self.BLUR_ITERATIONS = 2
        # basic (maximum) blurring value, range: [0, 1]
        self.BLUR_BASIC = 0.9
        # blurring decline factor, range: [1, inf)
        self.BLUR_DECLINE = 10.0
        # weight adaption heuristic interval (number of evaluations), range: [1, inf)
        self.WEIGHT_ADAPTION_DURATION = 50
        # small self.epsilon to avoid division by zero
        self.EPSILON = 0.0000001
        self.MAX_ITERATIONS = 1000
        self.MAX_RESETS = 100
        self.candidate_counter = 0
        
    def initialize_constants(self):
        """
        Initialize constants that depend on the instance.
        """
        # maximum pheromone value
        self.PH_MAX = np.float_( len(self.variables) / (1.0 - self.PH_REDUCE_FACTOR))
        # minimum pheromone value
        self.PH_MIN = np.float_(self.PH_MAX / (2*len(self.variables)) )

    def initialize_clause_weights(self):
        """
        Initializes clause weights for weight adaption duration. Initially
        all clauses are weighted equal.
        """
        self.clause_weights = np.ones(len(self.clauses), dtype=np.int)

    def initialize_mcv_heuristic(self):
        """
        Most Constrained Variable (MCV) heuristic: variables that appear in 
        most clauses are more important and visited more often.
        """
        shape = (len(self.clauses), 2, len(self.variables))
        self.truth_clauses = np.zeros(dtype=np.bool, shape=shape)
        for i, clause in enumerate(self.clauses):
            for lit in clause:
                if lit > 0:
                    self.truth_clauses[i][0][self.variables.index(lit)] = True
                else:
                    self.truth_clauses[i][1][self.variables.index(-lit)] = True
        self.int_clauses = self.truth_clauses.astype(int)
        self.mcv = np.sum(self.int_clauses, axis=0)

    def initialize_pheromones(self):
        """
        Initialize all pheromones to PH_MAX.
        """
        self.pheromones = np.ndarray((2, len(self.variables)), dtype=np.float)
        self.pheromones.fill(self.PH_MAX)

    def reset_pheromones(self):
        """
        Reset all pheromones.
        """
        np.random.seed()
        self.pheromones = np.random.uniform(low=self.PH_MIN, high=self.PH_MAX, size=((2, len(self.variables))) )

    def initialize_probabilities(self):
        """
        Create probabilities array and update probabilities with pheromones.
        """
        self.probabilities = np.ndarray((2, len(self.variables)), dtype=np.float)
        self.update_probabilities()

    def update_probabilities(self):
        """
        Update probabilities based on pheromones and MCV heuristic.
        """
        self.probabilities = self.pheromones**self.EXP_PH * self.mcv**self.EXP_MCV

    def choose_literals(self):
        """
        Choose num_vars literals, each with associated probability.
        """
        # reciprocal norm vector for probabilities of positive literals
        normalization_vector = (np.sum(self.probabilities, axis=0) + self.EPSILON) ** -1
        # for each variable decide wheter to take positive or negative literal
        np.random.seed()
        chosen = np.random.rand(len(self.variables)) < (normalization_vector * self.probabilities[0])
        return chosen

    def evaluate_solution(self, chosen):
        """
        Evaluate a solution candidate. Return quality of solution (with
        weight adaption heuristic) and number of solved clauses.
        """
        self.candidate_counter += 1

        # evaluation function in abstract superclass
        
        solved_clauses = np.any(self.truth_clauses & np.array([chosen, ~chosen]), axis=(2, 1)) 
        num_solved_clauses = np.sum(solved_clauses)
        # calculate evaluation with weight adaption heuristic
        evaluation = np.sum(solved_clauses * self.clause_weights)

        if self.candidate_counter == self.WEIGHT_ADAPTION_DURATION:
            # increase weights for unsatisfied clauses
            self.clause_weights += ~solved_clauses
            self.candidate_counter = 0

        return evaluation, num_solved_clauses

    def update_pheromones(self, chosen, evaluation):
        """
        Update pheromones based on solution candidate and its evaluation.
        """
        self.pheromones = self.pheromones * (1.0 - self.PH_REDUCE_FACTOR) + np.array([chosen, ~chosen]) * evaluation
        self.update_pheromones_bounds()

    def update_pheromones_bounds(self):
        """
        Make sure that pheromone values stay within PH_MIN and PH_MAX.
        """
        self.pheromones[self.pheromones < self.PH_MIN] = self.PH_MIN
        self.pheromones[self.pheromones > self.PH_MAX] = self.PH_MAX

    def blur_pheromones(self, max_divergence):
        """
        Blur pheromones by a random percental value within 
        [-max_divergence, max_divergence).
        """
        self.pheromones += self.pheromones * (np.random.rand(2, len(self.variables)) * max_divergence * 2 - max_divergence)
        self.update_pheromones_bounds()
        self.update_probabilities()

    def run(self):
        """
        Run the ant colony optimization algorithm.
        """
        self.initialize_constants()
        self.initialize_clause_weights()
        self.initialize_pheromones()
        self.initialize_mcv_heuristic()
        self.initialize_probabilities()
        for j in range(self.MAX_RESETS):
            for i in range(self.MAX_ITERATIONS):
                best_solution = None
                best_evaluation = -1
                best_num_solved = 0
                

                # simulate ant
                for a in range(self.NUM_ANTS):
                    # choose literals beased on probabilities
                    literals = self.choose_literals()
                    # print("Literals:", np.shape(literals))
                    # evaluate solution candidate
                    evaluation, num_solved_clauses = self.evaluate_solution(literals)
                    

                    if evaluation > best_evaluation:
                        # save best solution candidate
                        best_evaluation = evaluation
                        best_solution = literals 
                        best_num_solved = num_solved_clauses

                        if num_solved_clauses == len(self.clauses):
                            # solution was found
                            for i,truth_val in enumerate(literals):
                                self.truth_values[self.variables[i]] = truth_val
                            return True
                if i%10 == 0:
                    print("iteration number, Percentage of clauses satisfied:", i, 100*best_num_solved/len(self.clauses), '%')
                # update pheromones based on best solution candidate
                self.update_pheromones(best_solution, evaluation)
                # update probabilities
                self.update_probabilities()

                if i > 0 and i % self.BLUR_ITERATIONS == 0:
                    # blur pheromones based on how long the algorithm is already running (declining)
                    self.blur_pheromones(self.BLUR_BASIC * np.e**(-i/self.BLUR_DECLINE))
            self.initialize_pheromones()
            self.update_probabilities()
            print("Reset number:",j)



class gsat(sat_solvers):

    def __init__(self):
        super().__init__()

    def dict_builder(self):
        self.max_resets = 100
        self.max_mutations = 5
        self.epsilon = 0.05
        self.init_prob = 0.09

        self.dictionary_brackets = {}
        self.dictionary_logic = {}
        self.dictionary_clauses = {}
        self.dictionary_clausestruths = {}

        for var in self.variables:
            clauses_with_var = []
            if var not in self.dictionary_logic:
                self.dictionary_logic[var] = 0
            for i in range(len(self.clauses)):
                self.dictionary_clauses[i] = self.clauses[i]
                self.dictionary_clausestruths[i] = 0
                if var in list(np.abs(self.clauses[i])):
                    if var in self.clauses[i]:
                        clauses_with_var.append(i)
                    else:
                        clauses_with_var.append(-i)
            self.dictionary_brackets[var] = clauses_with_var
        return 


    def ls_initialization(self):
        np.random.seed()
        for key, value in self.dictionary_logic.items():
            self.dictionary_logic[key] = np.random.binomial(1, self.init_prob)
        return 


    def ls_initial_checker2(self):
        for key, val in self.dictionary_logic.items():
            for key2 in self.dictionary_brackets[key]:
                if (val == 1 and key2 >= 0) or (val == 0 and key2 <= 0):
                    self.dictionary_clausestruths[abs(key2)] = 1
                else:
                    pass
        return 

    def one_bracket_tester(self, clause, key_id):
        test = 0
        for val in clause:
            if np.abs(val) != key_id:
                if (val >= 0 and self.dictionary_logic[np.abs(val)] == 1) or (val <= 0 and self.dictionary_logic[np.abs(val)] == 0):
                    test = 1
            if np.abs(val) == key_id:
                if (val >= 0 and self.dictionary_logic[np.abs(val)] == 0) or (val <= 0 and self.dictionary_logic[np.abs(val)] == 1):
                    test = 1
        return test

    def ls_foresight2(self, key_id):
        change = 0
        for val in self.dictionary_brackets[key_id]:
            if self.dictionary_clausestruths[np.abs(val)] == 0:
                if (val >= 0 and self.dictionary_logic[key_id] == 0) or (val <=0 and self.dictionary_logic[key_id] == 1):
                    change += 1
            if self.dictionary_clausestruths[np.abs(val)] == 1:
                test = self.one_bracket_tester(self.dictionary_clauses[np.abs(val)], key_id)
                if test == 1:
                    change += 1
                else:
                    change -= 1
        return change

    def ls_foresight_iterator(self):
        changes = {}
        for key, val in self.dictionary_logic.items():
            change = self.ls_foresight2(key)
            changes[key] = change
        return changes
        
    def ls_e_greedy_mutation(self, changes):
        if self.epsilon != 0:
            if random.random() <= self.epsilon:
                pick = random.choice(list(self.dictionary_logic.keys()))
            else:
                max_val = max(changes.values())
                pick = random.choice([k for (k, v) in changes.items() if v == max_val])
        else:
            max_val = max(changes.values())
            pick = random.choice([k for (k, v) in changes.items() if v == max_val])
        return pick


    def ls_dict_updater(self, pick):
        self.dictionary_logic[pick] = 1 - self.dictionary_logic[pick]
        for val in self.dictionary_brackets[pick]:
            if self.dictionary_clausestruths[np.abs(val)] == 0:
                if (val >= 0 and self.dictionary_logic[pick] == 0) or (val <=0 and self.dictionary_logic[pick] == 1):
                    self.dictionary_clausestruths[np.abs(val)] == 1
            if self.dictionary_clausestruths[np.abs(val)] == 1:
                test = self.one_bracket_tester(self.dictionary_clauses[np.abs(val)], pick)
                if test == 0:
                    self.dictionary_clausestruths[np.abs(val)] == 0
        return

    def local_search(self):
        solution_found = False
        for i in range(self.max_resets):
            if solution_found == False:
                print('Reset number: ', i)
                self.ls_initialization()
                self.ls_initial_checker2()
                for j in range(self.max_mutations):
                    if j % 2 == 0:
                        print('Mutation number: ', j, '; Percentage of clauses satisfied:', (100 * sum(self.dictionary_clausestruths.values())/len(self.dictionary_clausestruths)), '%')
                    if solution_found == False:
                        changes = self.ls_foresight_iterator()
                        if max(changes.values()) < 0:
                            break
                        pick = self.ls_e_greedy_mutation(changes)
                        self.ls_dict_updater(pick)
                        if min(self.dictionary_clausestruths.values()) == 1:
                            solution_found = True
        return solution_found

    def run(self):
        self.dict_builder()
        for key, value in self.dictionary_logic.items():
            self.truth_values[key] = bool(value)
        return self.local_search()  
         




if __name__ == "__main__":

    heuristic = sys.argv[1]
    filename = sys.argv[2]

    if heuristic == '-S1':
        algo = dpll()
    elif heuristic == '-S2':
        algo = gsat()
    elif heuristic == '-S3':
        algo = aco()

    algo.parse(filename)
    SAT = algo.run()
    print("Satisfied status: ", SAT)
    if SAT == True:
        with open(str(filename+'.out'), 'w') as outfile: # parse the specific sudoku problems
            outfile.write('p cnf '+ str(len(algo.truth_values))+ ' ' +str(len(algo.truth_values))+ '\n')
            for key,value in algo.truth_values.items():
                if value == True:
                    outfile.write(str(key) + ' 0\n')
                if value == False:
                    outfile.write(str(-key) + ' 0\n')
    elif SAT == False:
        with open(str(filename+'.out'), 'w') as outfile: # parse the specific sudoku problems
            outfile.write(' ')
