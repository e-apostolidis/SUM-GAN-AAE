''' A Dynamic Programming based Python 
  Program for 0-1 Knapsack problem 
  Returns the maximum value that can 
  be put in a knapsack of capacity W '''

def knapSack(W, wt, val, n): 
	K = [[0 for x in range(W + 1)] for x in range(n + 1)] 

	# Build table K[][] in bottom up manner 
	for i in range(n + 1): 
		for w in range(W + 1): 
			if i == 0 or w == 0:
				K[i][w] = 0 
			elif wt[i-1] <= w: 
				K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]], K[i-1][w]) 
			else: 
				K[i][w] = K[i-1][w]

	selected = []
	w = W
	for i in range(n,0,-1):
		if K[i][w]!= K[i-1][w]:
			selected.insert(0,i-1)
			w -= wt[i-1]

	return selected 

# Driver program to test above function
'''val = [4,4,2,2,2,4]
wt =  [2,2,1,1,1,2]
W = 7
n = len(val)
selected = knapSack(W, wt, val, n)
print(selected)'''

# This code is contributed by Bhavya Jain 

