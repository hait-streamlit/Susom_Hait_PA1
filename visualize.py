import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import time, copy

from utils import read_maze, format_maze
from AIMA_structs import GridProblem

from algorithms.part_one import breadth_first_bfs, astar_search
from algorithms.part_two import multi_point_bfs, multi_point_astar, continuous_astar, a_star_star, cluster_star
from algorithms.part_three import biased_GBFS

class GUI:
    def __init__(self):
        self.init_board()
        self.animation_speed = 0.01
    
    def init_board(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal')

        plt.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(tick1On=False)

        fig.set_facecolor('none')
        ax.set_facecolor('none')

        self.fig = fig
        self.ax = ax

    def draw_square(self, x, y, size, color):
        rect = patches.Rectangle((x, y), size, size, linewidth=0, edgecolor=color, facecolor=color, zorder=1)
        self.ax.add_patch(rect)
    
    def draw_circle(self, x, y, size, color):
        circle = patches.Circle((x, y), size, linewidth=0, edgecolor=color, facecolor=color, zorder=2)
        self.ax.add_patch(circle)

    def draw_arrow(self, x, y, color, direction):
        arrow = None
        match direction:
            case (0, 1): # UP arrow
                arrow = patches.Arrow(x + 0.75, y, 0, 0.5, linewidth=0, edgecolor=color, facecolor=color, width=0.5, zorder=3)
            case (0, -1): # DOWN arrow
                arrow = patches.Arrow(x + 0.25, y + 1, 0, -0.5, linewidth=0, edgecolor=color, facecolor=color, width=0.5, zorder=3)
            case (1, 0): # RIGHT arrow
                arrow = patches.Arrow(x, y + 0.25, 0.5, 0, linewidth=0, edgecolor=color, facecolor=color, width=0.5, zorder=3)
            case (-1, 0): # LEFT arrow
                arrow = patches.Arrow(x + 1, y + 0.75, -0.5, 0, linewidth=0, edgecolor=color, facecolor=color, width=0.5, zorder=3)

        if arrow != None:
            self.ax.add_patch(arrow)

    # udpate the initial board with the start, goal, and walls of the maze
    def draw_board(self, rows, cols, start, goals, obstacles):
        self.ax.set_xticks(np.arange(cols + 1))
        self.ax.set_yticks(np.arange(rows + 1))
        self.ax.set_xlim(0, cols)
        self.ax.set_ylim(0, rows)

        for obstacle in obstacles:
            self.draw_square(obstacle[0], obstacle[1], 1, 'white')
        for goal in goals:
            self.draw_circle(goal[0] + 0.5, goal[1] + 0.5, 0.3, 'green') # draw goal point (green)
        self.draw_circle(start.state[0] + 0.5, start.state[1] + 0.5, 0.3, 'yellow') # draw start point (yellow)

        self.board.pyplot(self.fig)
    
    # utility function to fetches the maze from the layouts folder and formats the data
    def fetch_board_data(self):
        current_maze = list()
        try:
            if st.session_state.part == "Part 1":
                current_maze = read_maze(f"layouts/{self.maze.lower()}Maze.lay")
            elif st.session_state.part == "Part 2" or st.session_state.part == "Part 3":
                current_maze = read_maze(f"layouts/{self.maze.lower()}Search.lay")
        except Exception as e:
            if st.session_state.part == "Part 1":
                current_maze = read_maze(f"layouts/smallMaze.lay")
            elif st.session_state.part == "Part 2":
                current_maze = read_maze(f"layouts/tinySearch.lay")
            elif st.session_state.part == "Part 3":
                current_maze = read_maze(f"layouts/mediumMaze.lay")
        
        rows, cols = len(current_maze), len(current_maze[0]) - 1
        self.start, self.goals, self.obstacles = format_maze(current_maze)
        self.draw_board(rows, cols, self.start, self.goals, self.obstacles)

    # run the search and store the data so it can be animated
    def get_play_data(self):
        goals_copy = copy.deepcopy(self.goals)
        current_problem = GridProblem(self.start, goals_copy, self.obstacles)

        if st.session_state.part == "Part 1":
            if st.session_state.algorithm == "BFS":
                self.expanded_nodes, self.max_depth_searched, self.max_frontier_size, self.animation_sequence = breadth_first_bfs(current_problem)
            elif st.session_state.algorithm == "A*":
                self.expanded_nodes, self.max_depth_searched, self.max_frontier_size, self.animation_sequence = astar_search(current_problem)
        
        elif st.session_state.part == "Part 2":
            if st.session_state.algorithm == "BFS":
                self.expanded_nodes, self.max_depth_searched, self.max_frontier_size, self.animation_sequence = multi_point_bfs(current_problem)
            elif st.session_state.algorithm == "A*":
                self.expanded_nodes, self.max_depth_searched, self.max_frontier_size, self.animation_sequence = multi_point_astar(current_problem)
            elif st.session_state.algorithm == "C*":
                self.expanded_nodes, self.max_depth_searched, self.max_frontier_size, self.animation_sequence = continuous_astar(current_problem)
            elif st.session_state.algorithm == "A**":
                self.expanded_nodes, self.max_depth_searched, self.max_frontier_size, self.animation_sequence = a_star_star(current_problem)
            elif st.session_state.algorithm == "Clusters":
                cluster_size = {
                    "Tiny": 1.5,
                    "Small": 3,
                    "Tricky": 6,    
                }
                self.expanded_nodes, self.max_depth_searched, self.max_frontier_size, self.animation_sequence = cluster_star(current_problem, cluster_size[st.session_state.maze])

        elif st.session_state.part == "Part 3":
            if st.session_state.algorithm == "Biased GBFS":
                self.expanded_nodes, self.max_depth_searched, self.max_frontier_size, self.animation_sequence = biased_GBFS(current_problem)

    # draw the search path of the algorithm
    def animate_search_path(self):
        total_frames = sum(len(sequence) for sequence in self.animation_sequence)
        def generate_color(count, total_frames):
            r = int(180 * (count / total_frames)) + 75 if total_frames > 0 else 255
            return f'#{r:02X}AAAA'
        
        count = 0
        for sequence in self.animation_sequence:
            for frame in sequence:
                color = generate_color(count, total_frames)
                self.draw_square(frame.state[0], frame.state[1], 1, color)
                
                count += 1
                self.board.pyplot(self.fig)
                self.status.progress(count / total_frames)
            time.sleep(self.animation_speed)

    # reconstructs the optimal path
    # grabs the final node of each sequence and traverses back to the start node
    def get_optimal_path(self):
        full_sequence = []
        for sequence in reversed(self.animation_sequence):
            final_node = sequence[-1]
            
            # like a do while loop
            while True:
                full_sequence.append(final_node)
                if final_node.parent == None:
                    break
                else:
                    final_node = final_node.parent

        return full_sequence

    # draw the optimal path
    def animate_optimal_path(self):
        animation_sequence = self.get_optimal_path()

        total_frames = len(animation_sequence)
        def generate_color(count, total_frames): # incrementally alter the color of the path
            b = int(200 * (count / total_frames)) + 55 if total_frames > 0 else 255
            return f'#0000{b:02X}'
            
        count = 0
        for node in reversed(animation_sequence):
            color = generate_color(count, total_frames)
            self.draw_arrow(node.state[0], node.state[1], color, node.base_action)

            self.board.pyplot(self.fig)
            count += 1
            time.sleep(self.animation_speed)

    def draw_final_statistics(self):
        final_node = self.animation_sequence[-1][-1]
        df = pd.DataFrame(data={
            "Algorithm": [st.session_state.algorithm],
            "Maze Type": [st.session_state.maze],
            "Path Cost": [final_node.path_cost],
            "Expanded Nodes": [self.expanded_nodes],
            "Max Depth Searched": [self.max_depth_searched],
            "Max Frontier Size": [self.max_frontier_size]
        }, index=[0])
        
        st.table(df)
        with st.expander("Full Optimal Path"):
            def translate_action(action):
                match action:
                    case (0, 1):
                        return "UP"
                    case (0, -1):
                        return "DOWN"
                    case (1, 0):
                        return "RIGHT"
                    case (-1, 0):
                        return "LEFT"
                    case _:
                        return "START"
                        
            st.write("With the form (x, y) where the origin is the bottom left corner.")
            st.write("The x and y axes operate like a cartesian plane.")
            st.write("Implementations which restart the search from the start node will feature multiple START states corresponding to the algorithm reset.")
            for node in reversed(self.get_optimal_path()):
                st.write(node.state, translate_action(node.base_action))

    def run(self):
        st.title("PA1 - Search")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self.part = st.selectbox("Select assignment part", ["Part 1", "Part 2", "Part 3"], on_change=self.fetch_board_data, key="part")
        with col2:
            match st.session_state.part:
                case "Part 1":
                    self.algorithm = st.selectbox("Select an algorithm", ["BFS", "A*"], key="algorithm")
                case "Part 2":
                    self.algorithm = st.selectbox("Select an algorithm", ["BFS", "A*", "C*", "A**", "Clusters"], key="algorithm")
                case "Part 3":
                    self.algorithm = st.selectbox("Select an algorithm", ["Biased GBFS"], key="algorithm")
        with col3:
            current_maze_options = list()
            match st.session_state.part:
                case "Part 1":
                    current_maze_options = ["Small", "Medium", "Big", "Open"]
                case "Part 2":
                    current_maze_options = ["Tiny", "Small", "Tricky"]
                case "Part 3":
                    current_maze_options = ["Medium", "Big"]
            self.maze = st.selectbox("Select maze", current_maze_options, on_change=self.fetch_board_data, key="maze")

        self.board = st.pyplot(self.fig)
        self.fetch_board_data()

        self.status = st.empty()
        if st.button("Play"):
            try:
                self.get_play_data()

                self.animate_search_path()
                self.animate_optimal_path()
                self.draw_final_statistics() 
            except Exception as e:
                st.warning(f"No solution found for {st.session_state.algorithm} on {st.session_state.maze} maze")
            
        
        