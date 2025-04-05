from .part_two import multi_point_search
import math

def biased_GBFS(problem, h=None):
    original_center = problem.initial.state # track the original problem center
    def manhattan_distance(goal): # get the minimum manhattan distance to the goal
        current_min = math.inf
        current_bias = 0
        for goal in problem.goal: 
            current_min = min(current_min, abs(problem.initial.state[0] - goal[0]) + abs(problem.initial.state[1] - goal[1]))
            current_bias = min(current_bias, math.dist(original_center, goal))
        return current_min + current_bias # return the sum of the minimum manhattan distance and the bias
    
    h = h or manhattan_distance
    return multi_point_search(problem, f=lambda n: h(n))

