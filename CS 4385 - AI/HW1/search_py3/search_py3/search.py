# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #  (last-in, first-out)
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    stack = util.Stack()                # declare stack for a list
    start_position = problem.getStartState()

    stack.push((start_position,[],0))
    visted_list = {}
    
    while not stack.isEmpty():          # When start position is not a empty
        current_position = stack.pop()  # Get stete, direction, cost of current position
        if problem.isGoalState (current_position[0]): return current_position[1]    # check if the final then stop
        
        # add to visted list
        visted_list[current_position[0]] = True             # visted list is variable type, then can sace value at position 
        for state, direction, cost in problem.getSuccessors(current_position[0]):   # getSuccessors id check next 2 step thet the pacman can go
            if state not in visted_list:
                stack.push((state,current_position[1]+[direction],current_position[2]+cost))    # add to queue life DFS rule
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #  (First-in, first-out)
    queue = util.Queue()            # declare queue for a list
    start_position = problem.getStartState()

    queue.push((start_position,[],0))
    visted_list = {}

    while not queue.isEmpty():              # When start position is not a empty
        current_position = queue.pop()
        if problem.isGoalState (current_position[0]): return current_position[1]    # check if the final then stop

        if current_position[0] not in visted_list:         # check if state not visit then add to visted list
            visted_list[current_position[0]] = True             # visted list is variable type, then can sace value at position 
            for state, direction, cost in problem.getSuccessors(current_position[0]):   # getSuccessors id check next 2 step thet the pacman can go
                if state not in visted_list:
                    queue.push((state,current_position[1]+[direction],current_position[2]+cost))    # add to queue life BFS rule
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    priorityQueue = util.PriorityQueue()        # declare Priority Queue for a list
    start_position = problem.getStartState()
    
    priorityQueue.push((start_position,[],0),0)
    visted_list = {}

    while not priorityQueue.isEmpty():      # When start position is not a empty
        current_position = priorityQueue.pop()
        if problem.isGoalState (current_position[0]): return current_position[1]    # check if the final then stop

        if current_position[0] not in visted_list:  # check if state not visit then add to visted list
            visted_list[current_position[0]] = True             # visted list is variable type, then can sace value at position 
            for state, direction, cost in problem.getSuccessors(current_position[0]):   # getSuccessors id check next 2 step thet the pacman can go
                if state not in visted_list:
                    priorityQueue.push((state,current_position[1]+[direction],current_position[2]+cost),current_position[2]+cost)       # add to queue life BFS rule, but have the cost number, then we have UCS

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # heuristic use mahattanH euristic
    priorityQueue = util.PriorityQueue()        # declare Priority Queue for a list
    start_position = problem.getStartState()
    
    priorityQueue.push((start_position,[],0),0)
    visted_list = {}

    while not priorityQueue.isEmpty():      # When start position is not a empty
        current_position = priorityQueue.pop()
        if problem.isGoalState (current_position[0]): return current_position[1]    # check if the final then stop

        if current_position[0] not in visted_list:  # check if state not visit then add to visted list
            visted_list[current_position[0]] = True             # visted list is variable type, then can sace value at position 
            for state, direction, cost in problem.getSuccessors(current_position[0]):   # getSuccessors id check next 2 step thet the pacman can go
                if state not in visted_list:
                    priorityQueue.push((state,current_position[1]+[direction],current_position[2]+cost), current_position[2]+ cost + heuristic(state,problem))    # add to queue life BFS rule, but have the cost number + heuristic, then we have UCS


    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
