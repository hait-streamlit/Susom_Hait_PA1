from AIMA_structs import PriorityQueue, Node, GridProblem
from .part_one import g, expand
import math

def multi_point_bfs(problem):
    return multi_point_search(problem, f=len)

def multi_point_astar(problem, h=None):
    h = h or problem.h
    return multi_point_search(problem, f=lambda n: g(n) + h(n))

# utility derived from part one's best first search
def multi_point_search(problem, f):
    frontier = None
    reached = {}
    
    # use init_search as a function which can be called to reinitialize the search when a goal is reached
    def init_search(problem):
        nonlocal frontier, reached
        frontier = PriorityQueue([problem.initial], key=f)
        
        reached.clear()
        reached[problem.initial.state] = problem.initial

    max_frontier_size = 0
    expanded_nodes = 0
    max_depth_searched = 0
    full_sequence = list()
    
    #goals = copy.deepcopy(problem.goal)
    init_search(problem)
    current_sequence = []
    while frontier:
        # recalcualte max fringe size
        max_frontier_size = max(max_frontier_size, len(frontier))
        
        node = frontier.pop()
        current_sequence.append(node)

        max_depth_searched = max(max_depth_searched, node.depth)

        if node.state in problem.goal: # handle goal finding
            full_sequence.append(current_sequence)
            problem.goal.remove(node.state) # remove the goal from the list of goals
            
            if problem.goal != []: # if there are still goals to be reached, reset the search
                problem.initial = Node(node.state, path_cost=node.path_cost, depth=0)
                current_sequence = []
                init_search(problem)
            else: # if there are no more goals to be reached, return the sequence
                return expanded_nodes, max_depth_searched, max_frontier_size, full_sequence
        else:
            expanded_nodes += 1 # account for expanded nodes -> before expansion
            for child in expand(problem, node):
                s = child.state
                if s not in reached or child.path_cost < reached[s].path_cost:
                    reached[s] = child
                    frontier.add(child)
    
    raise Exception("No solution found")

def continuous_astar(problem, f=None):
    frontier = {} # use a dictionary to store the frontier

    if f is None:
        def f(n):
            def h(n): # use manhattan distance as the heuristic
                min_dist = float('inf')
                for goal in problem.goal: # iterate through all the goals and return the smallest distance
                    min_dist = min(min_dist, abs(n.state[0] - goal[0]) + abs(n.state[1] - goal[1]))
                return min_dist
            return g(n) + h(n)
    
    def init_search(problem):
        add_node(problem.initial)

    def add_node(node):
        frontier[node.state] = [node, f(node)]

    # new pop function for the new frontier implementation
    def pop_min():
        try:
            min_state = min(frontier, key=lambda s: frontier[s][1]) 
            min_node, min_val = frontier[min_state]

            del frontier[min_state]
            return min_node, min_val
        except Exception as e:
            raise Exception("No frontier")
    
    # penalize function to penalize the current node when a goal is reached
    def penalize_current():
        for node in frontier:
            frontier[node][1] *= 10 

    max_frontier_size = 0
    expanded_nodes = 0
    max_depth_searched = 0
    full_sequence = list()
    
    init_search(problem)
    current_sequence = []
    count = 0
    while frontier:
        count += 1
        if count > 1000: # break if the search is taking too long
            break

        # recalcualte max fringe size
        max_frontier_size = max(max_frontier_size, len(frontier))
        
        node, val = pop_min()
        current_sequence.append(node)

        max_depth_searched = max(max_depth_searched, node.depth)

        if node.state in problem.goal:
            problem.goal.remove(node.state)

            penalize_current() # penalize the current nodes when a goal is reached
            
            if problem.goal == []: # if there are no more goals to be reached, return the sequence (otherwise continue as normal)
                full_sequence.append(current_sequence)
                return expanded_nodes, max_depth_searched, max_frontier_size, full_sequence
            
        expanded_nodes += 1 # account for expanded nodes -> before expansion
        for child in expand(problem, node):
            s = child.state
            if child.state not in frontier:
                add_node(child)
            elif f(child) < frontier[child.state][1]:
                del frontier[child.state]
                add_node(child)
            
    raise Exception("No solution found")

def a_star_star(problem):
    heuristic_expansion = 0
    def h(n): # use A* as a heuristic
        min_dist = float('inf')
        for goal in problem.goal:
            sub_problem = GridProblem(Node(n.state), [goal], problem.obstacles)
            en, _, _, sequence = multi_point_astar(sub_problem) # use A* to find define the heuristic distance 
            min_dist = min(min_dist, sequence[0][-1].path_cost)

            nonlocal heuristic_expansion
            heuristic_expansion += en # track the number of sub expansions to get the total expanded nodes
        return min_dist

    # use continuous astar to find the optimal path
    expanded_nodes, max_depth_searched, max_frontier_size, sequence = continuous_astar(problem, f=lambda n: g(n) + h(n))
    return expanded_nodes + heuristic_expansion, max_depth_searched, max_frontier_size, sequence

# use k-medoids to cluster the goals and then use the cluster to find the optimal path
def cluster_star(problem, cluster_size=1.5):
    # get the distances of each goal to the search start
    def get_dist(start, goal):
        sub_problem = GridProblem(start, [goal], problem.obstacles)
        _, _, _, sequence = multi_point_astar(sub_problem)
        return sequence[0][-1].path_cost

    goal_dists = {
        goal: get_dist(problem.initial, goal) for goal in problem.goal
    }

    # use k-medoids to cluster the goals
    def k_medoids(goals):
        clusters = []
        visited = set()

        for goal in goals:
            if goal not in visited:
                cluster = { "total" : 0, "values": [], "distance": math.inf, "center": goal }
                for candidate in set(goals) - visited:
                    dist = math.dist(goal, candidate)
                    if dist < cluster_size: # define a variable cluster threshold (good for testing different values)
                        cluster["values"].append(candidate)
                        visited.add(candidate)
                        cluster["total"] += 1
                        if goal_dists[candidate] < cluster["distance"]:
                            cluster["distance"] = goal_dists[candidate]
                            cluster["center"] = candidate
                clusters.append(cluster)
        return clusters

    problem.goal = k_medoids(problem.goal)
    def h(n): # still use manhattan distance but now iterate through the clusters to find the smallest distance
        min_dist = float('inf')
        cluster_size = None
        for index, cluster in enumerate(problem.goal):
            if cluster["total"] > 0:
                dist = abs(n.state[0] - cluster["center"][0]) + abs(n.state[1] - cluster["center"][1])
                if dist < min_dist:
                    min_dist = dist
                cluster_size = cluster["total"]
        return min_dist * 2 / cluster_size # weight the distance by the cluster size

    return cluster_star_driver(problem, f=lambda n: g(n) + h(n), goal_dists=goal_dists)

# utility derived from part one's best first search
def cluster_star_driver(problem, f, goal_dists):
    frontier = None
    reached = {}
    
    # check if the node is a goal inside the clusters
    def is_goal(n):
        for cluster in problem.goal:
            if n.state in cluster["values"]:
                return True
        return False
    
    # update the cluster when a goal is reached
    def update_cluster(n):
        for cluster in problem.goal:
            if n.state in cluster["values"]:
                del cluster["values"][cluster["values"].index(n.state)]
                cluster["total"] -= 1
                if cluster["total"] > 0:
                    for element in cluster["values"]:
                        if goal_dists[element] < cluster["distance"]:
                            cluster["distance"] = goal_dists[element]
                            cluster["center"] = element
                else:
                    cluster["distance"] = math.inf

    def init_search(problem):
        nonlocal frontier, reached
        frontier = PriorityQueue([problem.initial], key=f)
        
        reached.clear()
        reached[problem.initial.state] = problem.initial

    max_frontier_size = 0
    expanded_nodes = 0
    max_depth_searched = 0
    full_sequence = list()
    
    #goals = copy.deepcopy(problem.goal)
    init_search(problem)
    current_sequence = []
    while frontier:
        # recalcualte max fringe size
        max_frontier_size = max(max_frontier_size, len(frontier))
        
        node = frontier.pop()
        current_sequence.append(node)

        max_depth_searched = max(max_depth_searched, node.depth)

        if is_goal(node):
            full_sequence.append(current_sequence)
            update_cluster(node) # update the cluster when a goal is reached
            
            if all(cluster["total"] == 0 for cluster in problem.goal):
                return expanded_nodes, max_depth_searched, max_frontier_size, full_sequence
            else:
                problem.initial = Node(node.state, path_cost=node.path_cost, depth=0)
                current_sequence = []
                init_search(problem)
                
        else:
            # expand the current node
            expanded_nodes += 1 # increment to account for expanding the current node
            for child in expand(problem, node):
                s = child.state
                if s not in reached or child.path_cost < reached[s].path_cost:
                    reached[s] = child
                    frontier.add(child)
    
    raise Exception("No solution found")
