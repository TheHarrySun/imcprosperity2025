
'''
manual trading: round 1
'''

import numpy as np

'''
say: 
- index 0 is snowball
- index 1 is pizza
- index 2 is silicon nugget
- index 3 is seashells
in the exchange matrix, exchange[i][j] is the exchange rate for one of i to j
'''
prods = ['snowball', 'pizza', 'silicon nugget', 'seashell']

exchange = []
snowball_exchanges = [1, 1.45, 0.52, 0.72]
pizza_exchanges = [0.7, 1, 0.31, 0.48]
nuggets_exchanges = [1.95, 3.1, 1, 1.49]
seashells_exchanges = [1.34, 1.98, 0.64, 1]
exchange.append(snowball_exchanges)
exchange.append(pizza_exchanges)
exchange.append(nuggets_exchanges)
exchange.append(seashells_exchanges)
exchange = np.array(exchange)

MAX_TRADES = 5
NUM_PRODS = 4
max_array = np.zeros((MAX_TRADES, NUM_PRODS))
back_track = np.zeros((MAX_TRADES, NUM_PRODS))
for i in range(0, MAX_TRADES):
    for j in range(0, NUM_PRODS):
        if (i == 0):
            max_array[i, j] = 500 * exchange[3, j]
            back_track[i, j] = -1
        elif (i == 4 and j != 3):
            back_track[i, j] = -1
        else:
            for k in range(0, NUM_PRODS):
                if (max_array[i, j] < max_array[i-1, k] * exchange[k, j]):                
                    max_array[i, j] = max_array[i - 1, k] * exchange[k, j]
                    back_track[i, j] = k
    
print(max_array)
print(back_track)
backtrack_seq = []                
for i in range(MAX_TRADES - 1, -1, -1):
    if (i == MAX_TRADES - 1):
        backtrack_seq.append(back_track[i, 3])
    else:
        backtrack_seq.append(back_track[i, int(backtrack_seq[-1])])
        
        
for i in range(len(backtrack_seq) - 1, -1, -1):
    print(backtrack_seq[i])
    
for i in range(len(backtrack_seq) - 1, -1, -1):
    if (backtrack_seq[i] == -1):
        print("seashells -> ", end="")
    else:
        print(prods[int(backtrack_seq[i])], "-> ", end = "")
print("seashells")