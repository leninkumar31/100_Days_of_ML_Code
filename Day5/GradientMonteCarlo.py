import numpy as np
class RandomWalk(object):
    def __init__(self,numStates,groupSize):
        self.numStates = numStates
        self.currState = numStates/2
        self.leftTerminal = 0
        self.rightTerminal = numStates + 1
        self.groupSize = groupSize
        self.Terminal = False

    def isTerminal(self):
        if self.currState==self.leftTerminal:
            self.Terminal = True
            return -1
        elif self.currState==self.rightTerminal:
            self.Terminal = True
            return 1
        return 0

    def make_action(self,state):
        leftmostState = max(0,state-50)
        rightmostState = min(state+50,1001)
        total = rightmostState - leftmostState
        possible_states = []
        equalProb = 0.01
        prob = np.ones(total, dtype=float) * equalProb
        for i in range(leftmostState,rightmostState+1,1):
            if i !=state:
                possible_states.append(i)
        if total!=100:
            if state<50:
                prob[0] = (50-state + 1) * equalProb
            elif state>950:
                prob[total-1] = (50-(1001-state)+1) * equalProb
        return np.random.choice(a=possible_states, p=prob)

    def get_next(self,state):
        self.currState = self.make_action(state)
        return self.currState,self.isTerminal()

alpha = 2 * 0.00001
episodes = 100000
weights = np.zeros(10,dtype=float)
for episodeCnt in range(episodes):
    print episodeCnt
    env = RandomWalk(1000,100)
    episode = []
    current_state = env.currState
    while not env.Terminal:
        next_state,reward = env.get_next(current_state)
        episode.append((current_state,reward))
        current_state = next_state
    for i,(state,reward) in enumerate(episode):
        G = sum([val for (_,val) in episode[i:]])
        weights[(state-1)/100] += alpha * (G - weights[(state-1)/100])
value_function = np.zeros(1001,dtype=float)
for i in range(1,1001,1):
    value_function[i] = weights[(i-1)/100]

import matplotlib.pyplot as plt
plt.plot(range(1,1001),value_function[1:])
plt.xlabel("states")
plt.ylabel("Value scale")
plt.show()