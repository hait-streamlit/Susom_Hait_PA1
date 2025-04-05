from AIMA_structs import Node, PriorityQueue

# breadth first search implementation
def breadth_first_bfs(problem):
    "Search shallowest nodes in the search tree first; using best-first."
    expanded_nodes, max_depth_searched, max_frontier_size, sequence = best_first_search(problem, f=len)
    return expanded_nodes, max_depth_searched, max_frontier_size, [sequence]

# get the path cost from the node itself
def g(n): return n.path_cost

def astar_search(problem, h=None):
    """Search nodes with minimum f(n) = g(n) + h(n)."""
    h = h or problem.h
    expanded_nodes, max_depth_searched, max_frontier_size, sequence = best_first_search(problem, f=lambda n: g(n) + h(n))
    return expanded_nodes, max_depth_searched, max_frontier_size, [sequence]

# best first search is a utility to breadth first search
def best_first_search(problem, f):
    "Search nodes with minimum f(node) value first."
    frontier = PriorityQueue([problem.initial], key=f)
    reached = {problem.initial.state: problem.initial}

    # added the following variables to track the search
    max_frontier_size = 1
    expanded_nodes = 0
    max_depth_searched = 0
    sequence = list()
    
    while frontier:
        node = frontier.pop()
        sequence.append(node)

        if problem.is_goal(node.state): # if the node is a goal, return the sequence
            return expanded_nodes, max_depth_searched, max_frontier_size, sequence
        
        # add to expanded nodes right before expansion
        expanded_nodes += 1
        for child in expand(problem, node): 
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)

        # recalcualte max fringe size
        if (len(frontier) > max_frontier_size): 
            max_frontier_size = len(frontier)
        if (node.depth > max_depth_searched): max_depth_searched = node.depth
    
    raise Exception("No solution found")

# expand definition
def expand(problem, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        depth = node.depth + 1
        yield Node(state=s1, parent=node, action=action, base_action=(action[0] - s[0], action[1] - s[1]), path_cost=cost, depth=depth)