from AIMA_structs import Node

# read the maze file and return a 2d array containing the maze
def read_maze(file_path):
    with open(file_path, 'r') as maze_file:
        maze = [[element for element in line] for line in maze_file]
        return maze

# print the input maze (input: 2d array)
def print_maze(maze):
    for row in maze:
        print("".join(row)) 

def format_maze(maze):
    start, goal, obstacles = (), [], []
    
    for y, row in enumerate(maze):
        for x, index in enumerate(row):
            if index == "P":
                start = (x, y)
            elif index == "%":
                obstacles.append((x, y))
            elif index == ".":
                goal.append((x, y))
    
    return Node(start), goal, tuple(obstacles)