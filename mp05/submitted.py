# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)
import queue

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    paths = {}
    q = queue.SimpleQueue()
    explored = set()
    start = maze.start
    if start in maze.waypoints:
        return [start]
    q.put(start)
    frontier = {start}
    paths[start] = [start]
    while not q.empty():
        node = q.get()
        frontier.remove(node)
        explored.add(node)
        path_to_node = paths[node]
        neighbors = maze.neighbors(node[0], node[1])
        for neighbor in neighbors:
            if neighbor in explored or neighbor in frontier:
                continue
            paths[neighbor] = path_to_node + [neighbor]
            if neighbor in maze.waypoints:
                return paths[neighbor]
            q.put(neighbor)
            frontier.add(neighbor)

    return []

def manhattan_distance(src, dst):
    return abs(src[0] - dst[0]) + abs(src[1] - dst[1])

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    paths = {}
    q = queue.PriorityQueue()
    explored = set()
    start = maze.start
    if start in maze.waypoints:
        return [start]
    waypoint = maze.waypoints[0]
    start_prio = manhattan_distance(start, waypoint)
    q.put((start_prio, start))
    frontier = {start}
    paths[start] = [start]
    while not q.empty():
        prio, node = q.get()
        frontier.remove(node)
        explored.add(node)
        path_to_node = paths[node]
        neighbors = maze.neighbors(node[0], node[1])
        for neighbor in neighbors:
            if neighbor in explored or neighbor in frontier:
                continue
            paths[neighbor] = path_to_node + [neighbor]
            if neighbor in maze.waypoints:
                return paths[neighbor]
            q.put((len(paths[neighbor]) - 1 + manhattan_distance(neighbor, waypoint), neighbor))
            frontier.add(neighbor)

    return []

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    return []
