# coding=utf-8

import pandas as pd
import numpy as np
ACTIONS = [3, 4, 5]
INDEX = [1, 2]
DATA = [[1, 2, 3], [4, 5, 6]]
q_table = pd.DataFrame(
    data=DATA,
    index=INDEX,
    columns=ACTIONS
)
print(q_table)

print("------------")
print(q_table.iloc[1, ].max())
print(q_table.ix[1, 3])

print(np.random.random_integers(5))
a = 5; c=6
print(a/c, a//c)

def decode_state(state):
    return (state//24, state%24)

print(decode_state(28))

from CartPole_learning import *
print(lisan_angle(-0.00139605800602))