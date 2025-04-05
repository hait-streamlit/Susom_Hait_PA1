import math, heapq

# node definition
class Node:
    "A Node in a search tree."
    # added depth track
    def __init__(self, state, parent=None, action=None, base_action=None, path_cost=0, depth=0):
        self.__dict__.update(state=state, parent=parent, action=action, base_action=base_action, path_cost=path_cost, depth=depth)

    def __repr__(self): return '<{}>'.format(self.state)
    def __len__(self): return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): return self.path_cost < other.path_cost

# problem definition
class Problem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When yiou create an instance of a subclass, specify `initial`, and `goal` states 
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, initial=None, goal=None, **kwds): 
        self.__dict__.update(initial=initial, goal=goal, **kwds) 
        
    def actions(self, state):        raise NotImplementedError
    def result(self, state, action): raise NotImplementedError
    def is_goal(self, state):        return state in self.goal
    def action_cost(self, s, a, s1): return 1
    def h(self, node):               return 0
    
    def __str__(self):
        return '{}({!r}, {!r})'.format(
            type(self).__name__, self.initial, self.goal)

# manhattan distance utility for GridProblem h(n) (NEW)
def manhattan_distance(start, goals):
    current_min = math.inf
    for goal in goals: # iterate through all the goals and return the smallest distance
        current_min = min(current_min, abs(start[0] - goal[0]) + abs(start[1] - goal[1]))
    return current_min

# extends problem class to work on grid style problems (in this case a maze)
class GridProblem(Problem):
    """Finding a path on a 2D grid with obstacles. Obstacles are (x, y) cells."""

    def __init__(self, initial=(15, 30), goal=(130, 30), obstacles=(), **kwds):
        Problem.__init__(self, initial=initial, goal=goal, obstacles=set(obstacles), **kwds)

    # directions modified to include only UP, DOWN, LEFT, and RIGHT
    directions = [          (0, -1),
                  (-1, 0),           (1,  0),
                            (0, +1)         ]
    
    def action_cost(self, s, action, s1): return 1 # each action is unit cost in PA1
    
    def h(self, node): return manhattan_distance(node.state, self.goal)
                  
    def result(self, state, action): 
        "Both states and actions are represented by (x, y) pairs."
        return action if action not in self.obstacles else state
    
    def actions(self, state):
        """You can move one cell in any of `directions` to a non-obstacle cell."""
        x, y = state
        return {(x + dx, y + dy) for (dx, dy) in self.directions} - self.obstacles

# priority queue implementation (using a MIN heap)
class PriorityQueue:
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x): 
        self.key = key
        self.items = [] # a heap of (score, item) pairs
        for item in items:
            self.add(item)
         
    def add(self, item):
        """Add item to the queuez."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]
    
    def top(self): return self.items[0][1]

    def __len__(self): return len(self.items)
