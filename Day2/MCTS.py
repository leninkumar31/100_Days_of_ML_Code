import numpy as np
import Node

def softmax(x):
    z = np.exp(x)
    return z/np.sum(z)

def UppperConfidenceBound(node,c):
    ucbValues = np.zeros(len(node.child))
    for i in range(len(node.child)):
        ucbValues[i] = ((node.child[i].TotalSimualtionReward*1.0)/(node.child[i].TotalNumVisit+1))\
                    + c * np.sqrt(np.log(node.TotalNumVisit)/(node.child[i].TotalNumVisit+1))
    return node.child[np.random.choice(range(len(node.child)), p=softmax(ucbValues))]

def RolloutPolicy(node):
    unvisitedChild = []
    if not node.visited:
        node.visited = True
        node.child = node.get_child()
        unvisitedChild = node.child
    else:
        for i in range(len(node.child)):
            if not node.child[i].visited:
                unvisitedChild.append(node.child[i])
    # if len(unvisitedChild)==0:
    #     print node.TicTacToe.print_board()
    #     for i in range(len(node.child)):
    #         print node.child[i].TicTacToe.print_board()
    return unvisitedChild[np.random.choice(range(len(unvisitedChild)))]

def traverse(node,c):
    while node.expanded:
        node = UppperConfidenceBound(node,c)
    return RolloutPolicy(node)

def result(node):
    if node.TicTacToe.moveCnt & 1:
        return 1
    return 0

# def result(node):
#     if node.TicTacToe.moveCnt & 1:
#         if node.TicTacToe.draw:
#             return 0
#         else:
#             return 1
#     return 0

def simulation(node):
    while not node.Terminal:
        # print node.TicTacToe.print_board()
        node = RolloutPolicy(node)
    return result(node)

def backpropagation(node,result):
    if node is None:
        return
    node.TotalNumVisit += 1
    node.TotalSimualtionReward += result
    isExpanded = True
    for i in range(len(node.child)):
        if node.child[i].visited!=True:
            isExpanded = False
    node.expanded = isExpanded
    backpropagation(node.parent,result)

def pick_action(node):
    numVisArr = []
    for i in range(len(node.child)):
        numVisArr.append(node.child[i].TotalNumVisit)
    return node.child[np.random.choice(range(len(node.child)), p=softmax(numVisArr))]

def MonteCarloTreeSearch(root,c):
    for epoch in range(100):
        leaf = traverse(root,c) #leaf is node which is not fully expanded
        result = simulation(leaf)
        backpropagation(leaf,result)
    # print(root.TicTacToe.print_board())
    # for i in range(len(root.child)):
    #     root.child[i].TicTacToe.print_board()
    #     print root.child[i].TotalSimualtionReward
    #     print root.child[i].TotalNumVisit
    return pick_action(root)

