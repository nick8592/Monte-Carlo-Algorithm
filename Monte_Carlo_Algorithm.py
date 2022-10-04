# Problem: A robot needs to go from Start to Target
# Reward: +100 for target, -1 for each other step
# Q-value: for simplicity zero
# Initial Q values are all zero

from operator import index
from matplotlib.pyplot import grid
import matplotlib.pyplot as plt
import numpy as np
import random
np.random.seed(9527)
random.seed(9527)

MAX_ITERATION = 15000
MAX_EPISODE_LENGTH = 10
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1

class Environment():
    def __init__(self):
        self.rows = 4
        self.cols = 6 
        self.grid_world = [[  "T",  "s1",  "s2",  "s3",  "s4",  "s5"],
                           [ "s6",  "s7",  "s8",  "s9",   "W", "s10"],
                           ["s11",   "W", "s12",   "W", "s13", "s14"],
                           ["s15", "s16", "s17", "s18", "s19", "s20"]] #T: Target, W: Wall
        self.action_to_number = {"up": 0, "right":1, "down":2, "left":3}
        self.action_dict = {"up": [-1,0], "right": [0, 1], "down": [1,0], "left":[0,-1]}
        self.direction_dict = {0: "up", 1:"right", 2:"down", 3:"left"}
        self.invalid_start = ["T", "W"]

    def transfer_state(self, state_coordinates, action): #Input(state, action), Output(next state)
        current_state_coordinates = state_coordinates
        next_state_coordinates = state_coordinates + self.action_dict[action]

        if next_state_coordinates[0] < 0 or next_state_coordinates[0] > 3 or next_state_coordinates[1] < 0 or next_state_coordinates[1] > 5:# Out of board
            return current_state_coordinates
        next_state = self.grid_world[next_state_coordinates[0]][next_state_coordinates[1]]
        if next_state == "W": #Hit the wall 
            return current_state_coordinates
        return next_state_coordinates

class MC_Exploring_Start():
    def __init__(self):
        self.env = Environment()
        self.Max_iteration = MAX_ITERATION
        self.gamma = DISCOUNT_FACTOR
        self.Horizon = MAX_EPISODE_LENGTH #Max episode_length
        self.sum_of_Q_values = [0 for i in range(self.Max_iteration)]

        self.Q_values = {}
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] not in self.env.invalid_start:
                    for act in self.env.action_dict.keys():
                        self.Q_values[ ((row, col), act) ] = 0 #Initialize Q value (state, action) 

        self.returns_dict = {}
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] not in self.env.invalid_start:
                    for act in self.env.action_dict.keys():
                        self.returns_dict[ ((row, col), act) ] = [0, 0] #[Mean value, Visited count]

    def generate_initial_state(self): #Generate random state
        while True:
            state_row = np.random.randint(self.env.rows)
            state_col = np.random.randint(self.env.cols)
            if self.env.grid_world[state_row][state_col] in self.env.invalid_start:
                continue
            else:
                break
        return np.array([state_row, state_col])

    def generate_random_action(self):
        action = self.env.direction_dict[np.random.randint(4)]
        while (self.env.transfer_state(self.current_state_coordinates, action) == self.current_state_coordinates).all():
                action = self.env.direction_dict[np.random.randint(4)]
        return action

    def policy(self, state_coordinate): #Optimal policy, find the maximun Q(s,a) and return action  
        Q_value = []
        valid_actions = []
        state_coordinate = np.array(state_coordinate)
        indexes = []
        for action in self.env.action_dict.keys():
            # if state coordinate == next state coordinate, which means the action is invalid cause "transfer_state" return the same coordinate
            if not (state_coordinate == self.env.transfer_state(np.array(state_coordinate), action)).all():
                valid_actions.append(action)

        for valid_action in valid_actions:
            Q_value.append(self.Q_values[(tuple(state_coordinate), valid_action)])
        max_value = max(Q_value)

        for valid_action in valid_actions:
            if max_value == self.Q_values[(tuple(state_coordinate), valid_action)]:
                indexes.append(self.env.action_to_number[valid_action])
        return self.env.direction_dict[random.choice(indexes)]

    def print_Qvalue(self, state_coordinate): #For debugging
        Q_value = []
        for action in self.env.action_dict.keys():
            Q_value.append(self.Q_values[(tuple(state_coordinate), action)])
        return Q_value

    def iter(self): #Main loop
        for iterration in range(self.Max_iteration):
            episode = []
            sum_of_Q_value = 0
            self.current_state_coordinates = self.generate_initial_state()
            action = self.generate_random_action()
            for h in range(self.Horizon): #Generate episode
                next_state_coordinates = self.env.transfer_state(self.current_state_coordinates, action)
                reward = -1
                if self.env.grid_world[next_state_coordinates[0]][next_state_coordinates[1]] == "T":
                    reward = 100
                    episode.append([self.current_state_coordinates, action, reward])
                    break
                episode.append([self.current_state_coordinates, action, reward]) #Episode [[[coordinate],action, reward]]
                self.current_state_coordinates = next_state_coordinates
                action = self.policy(self.current_state_coordinates)
            G = 0
            
            for h in range(len(episode)-1, -1, -1): #Iterate through H-1, H-2,...,0
                coordinate, action, reward = episode[h]
                G = self.gamma*G + reward
                returns = self.returns_dict[(tuple(coordinate), action)]
                mean = returns[0]
                visited_count = returns[1]
                mean = (mean*visited_count + G)/(visited_count + 1)
                visited_count += 1
                self.returns_dict[(tuple(episode[h][0] ), episode[h][1])] = [mean, visited_count]
                self.Q_values[(tuple(episode[h][0]), episode[h][1])] = self.returns_dict[(tuple(episode[h][0]), episode[h][1])][0]
                # print(self.Q_values)


            # calculate sum of Q value according to current policy
            for row in range(self.env.rows):
                for col in range(self.env.cols):
                    if self.env.grid_world[row][col] not in self.env.invalid_start:
                        action = self.policy((row,col))
                        sum_of_Q_value += self.Q_values[((row, col), action)]
                        self.sum_of_Q_values[iterration] = sum_of_Q_value
            # print(f"{iterration}. sum of Q values: {self.sum_of_Q_values[iterration]}")


    def render(self): #Show results
        output = self.env.grid_world
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] in self.env.invalid_start:
                    continue
                else:
                    action = self.policy((row,col))
                    output[row][col] = action

        for row in range(0, self.env.rows):
            print("-------------------------------------------------------")
            out = "| "
            for col in range(0, self.env.cols):
                out += str(output[row][col]).ljust(6) + " | "
            print(out)
        print("-------------------------------------------------------")

class MC_Epsilon_Greedy():
    def __init__(self):
        self.env = Environment()
        self.Max_iteration = MAX_ITERATION
        self.gamma = DISCOUNT_FACTOR
        self.Horizon = MAX_EPISODE_LENGTH #Max episode_length
        self.sum_of_Q_values = [0 for i in range(self.Max_iteration)]

        self.Q_values = {}
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] not in self.env.invalid_start:
                    for act in self.env.action_dict.keys():
                        self.Q_values[ ((row, col), act) ] = 0 #Initialize Q value (state, action) 

        self.returns_dict = {}
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] not in self.env.invalid_start:
                    for act in self.env.action_dict.keys():
                        self.returns_dict[ ((row, col), act) ] = [0, 0] #[Mean value, Visited count]

    def generate_initial_state(self): #Generate random state
        while True:
            state_row = np.random.randint(self.env.rows)
            state_col = np.random.randint(self.env.cols)
            if self.env.grid_world[state_row][state_col] in self.env.invalid_start:
                continue
            else:
                break
        return np.array([state_row, state_col])

    def generate_random_action(self):
        action = self.env.direction_dict[np.random.randint(4)]
        while (self.env.transfer_state(self.current_state_coordinates, action) == self.current_state_coordinates).all():
                action = self.env.direction_dict[np.random.randint(4)]
        return action

    def policy(self, state_coordinate): #Optimal policy, find the maximun Q(s,a) and return action  
        Q_value = []
        valid_actions = []
        state_coordinate = np.array(state_coordinate)
        indexes = []
        for action in self.env.action_dict.keys():
            # if state coordinate == next state coordinate, \
            # which means the action is invalid cause "transfer_state" return the same coordinate
            if not (state_coordinate == self.env.transfer_state(state_coordinate, action)).all(): 
                valid_actions.append(action)

        if np.random.random() > EPSILON: # exploit
            for valid_action in valid_actions:
                Q_value.append(self.Q_values[(tuple(state_coordinate), valid_action)])
            max_value = max(Q_value)
            
            for valid_action in valid_actions:
                if max_value == self.Q_values[(tuple(state_coordinate), valid_action)]:
                    indexes.append(self.env.action_to_number[valid_action])
            return self.env.direction_dict[random.choice(indexes)]
        else: # explore
            for valid_action in valid_actions:
                indexes.append(self.env.action_to_number[valid_action])
            return self.env.direction_dict[random.choice(indexes)]


    def print_Qvalue(self, state_coordinate): #For debugging
        Q_value = []
        for action in self.env.action_dict.keys():
            Q_value.append(self.Q_values[(tuple(state_coordinate), action)])
        return Q_value

    def iter(self): #Main loop
        for iterration in range(self.Max_iteration):
            episode = []
            sum_of_Q_value = 0
            self.current_state_coordinates = self.generate_initial_state()
            action = self.generate_random_action()
            for h in range(self.Horizon): #Generate episode
                next_state_coordinates = self.env.transfer_state(self.current_state_coordinates, action)
                reward = -1
                if self.env.grid_world[next_state_coordinates[0]][next_state_coordinates[1]] == "T":
                    reward = 100
                    episode.append([self.current_state_coordinates, action, reward])
                    break
                episode.append([self.current_state_coordinates, action, reward]) #Episode [[[coordinate],action, reward]]
                self.current_state_coordinates = next_state_coordinates
                action = self.policy(self.current_state_coordinates)
            G = 0
            
            for h in range(len(episode)-1, -1, -1): #Iterate through H-1, H-2,...,0
                coordinate, action, reward = episode[h]
                G = self.gamma*G + reward
                returns = self.returns_dict[(tuple(coordinate), action)]
                mean = returns[0]
                visited_count = returns[1]
                mean = (mean*visited_count + G)/(visited_count + 1)
                visited_count += 1
                self.returns_dict[(tuple(episode[h][0] ), episode[h][1])] = [mean, visited_count]
                self.Q_values[(tuple(episode[h][0]), episode[h][1])] = self.returns_dict[(tuple(episode[h][0]), episode[h][1])][0]
                # print(self.Q_values)


            # calculate sum of Q value according to current policy
            for row in range(self.env.rows):
                for col in range(self.env.cols):
                    if self.env.grid_world[row][col] not in self.env.invalid_start:
                        action = self.policy((row,col))
                        sum_of_Q_value += self.Q_values[((row, col), action)]
                        self.sum_of_Q_values[iterration] = sum_of_Q_value
            # print(f"{iterration}. sum of Q values: {self.sum_of_Q_values[iterration]}")


    def render(self): #Show results
        output = self.env.grid_world
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] in self.env.invalid_start:
                    continue
                else:
                    action = self.policy((row,col))
                    output[row][col] = action

        for row in range(0, self.env.rows):
            print("-------------------------------------------------------")
            out = "| "
            for col in range(0, self.env.cols):
                out += str(output[row][col]).ljust(6) + " | "
            print(out)
        print("-------------------------------------------------------")


if __name__ == "__main__":
    print("Find optimal policy using Monte Carlo algorithm with exploring starts")
    monte = MC_Exploring_Start()
    monte_greedy = MC_Epsilon_Greedy()
    print("Before (random policy), T = Target, W = Wall")
    monte.render()

    # MC Exploring Start
    monte.iter()
    print("\nMC Exploring Start")
    print("\nOptimal policy, T = Target, W = Wall")
    monte.render()

    # MC Epsilon Greedy
    monte_greedy.iter()
    print("\nMC Epsilon Greedy")
    print("\nOptimal policy, T = Target, W = Wall")
    monte_greedy.render()
    print("\n\n\n")

    # plot compare
    x_axis = [i for i in range(monte.Max_iteration)]
    y_axis_MC_Exploring_Start = monte.sum_of_Q_values
    y_axis_MC_Epsilon_Greedy = monte_greedy.sum_of_Q_values

    plt.plot(x_axis, y_axis_MC_Exploring_Start, label = "MC Exploring Start")
    plt.plot(x_axis, y_axis_MC_Epsilon_Greedy, label = "MC Epsilon Greedy")

    text = f"Epsilon: {EPSILON} \nMax Iteration: {MAX_ITERATION}"
    plt.figtext(.8, .2, text)

    plt.xlabel('Iterations')
    plt.ylabel('Sum of Q Values')
    plt.title('Monte Carlo Algorithm')
    plt.legend()
    plt.show()   
    








