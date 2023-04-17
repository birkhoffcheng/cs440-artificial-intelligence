'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def is_wall(model, r, c):
	if r < 0 or r >= model.M or c < 0 or c >= model.N:
		return True
	return model.W[r, c]

def compute_transition_matrix(model):
	'''
	Parameters:
	model - the MDP model returned by load_MDP()

	Output:
	P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
	'''
	M = model.M
	N = model.N
	P = np.zeros((M, N, 4, M, N))
	for r in range(M):
		for c in range(N):
			if model.T[r, c]:
				continue

			# action = left
			if is_wall(model, r, c - 1):
				P[r, c, 0, r, c] += model.D[r, c, 0] # stays in place
			else:
				P[r, c, 0, r, c - 1] += model.D[r, c, 0] # goes left
			if is_wall(model, r + 1, c):
				P[r, c, 0, r, c] += model.D[r, c, 1] # stays in place
			else:
				P[r, c, 0, r + 1, c] += model.D[r, c, 1] # goes down
			if is_wall(model, r - 1, c):
				P[r, c, 0, r, c] += model.D[r, c, 2] # stays in place
			else:
				P[r, c, 0, r - 1, c] += model.D[r, c, 2] # goes up

			# action = up
			if is_wall(model, r - 1, c):
				P[r, c, 1, r, c] += model.D[r, c, 0] # stays in place
			else:
				P[r, c, 1, r - 1, c] += model.D[r, c, 0] # goes up
			if is_wall(model, r, c - 1):
				P[r, c, 1, r, c] += model.D[r, c, 1] # stays in place
			else:
				P[r, c, 1, r, c - 1] += model.D[r, c, 1] # goes left
			if is_wall(model, r, c + 1):
				P[r, c, 1, r, c] += model.D[r, c, 2] # stays in place
			else:
				P[r, c, 1, r, c + 1] += model.D[r, c, 2] # goes right

			# action = right
			if is_wall(model, r, c + 1):
				P[r, c, 2, r, c] += model.D[r, c, 0] # stays in place
			else:
				P[r, c, 2, r, c + 1] += model.D[r, c, 0] # goes right
			if is_wall(model, r - 1, c):
				P[r, c, 2, r, c] += model.D[r, c, 1] # stays in place
			else:
				P[r, c, 2, r - 1, c] += model.D[r, c, 1] # goes up
			if is_wall(model, r + 1, c):
				P[r, c, 2, r, c] += model.D[r, c, 2] # stays in place
			else:
				P[r, c, 2, r + 1, c] += model.D[r, c, 2] # goes down

			# action = down
			if is_wall(model, r + 1, c):
				P[r, c, 3, r, c] += model.D[r, c, 0] # stays in place
			else:
				P[r, c, 3, r + 1, c] += model.D[r, c, 0] # goes down
			if is_wall(model, r, c + 1):
				P[r, c, 3, r, c] += model.D[r, c, 1] # stays in place
			else:
				P[r, c, 3, r, c + 1] += model.D[r, c, 1] # goes right
			if is_wall(model, r, c - 1):
				P[r, c, 3, r, c] += model.D[r, c, 2] # stays in place
			else:
				P[r, c, 3, r, c - 1] += model.D[r, c, 2] # goes left

	return P

def update_utility(model, P, U_current):
	'''
	Parameters:
	model - The MDP model returned by load_MDP()
	P - The precomputed transition matrix returned by compute_transition_matrix()
	U_current - The current utility function, which is an M x N array

	Output:
	U_next - The updated utility function, which is an M x N array
	'''
	M = model.M
	N = model.N
	U_next = np.zeros((M, N))
	for r in range(M):
		for c in range(N):
			if model.T[r, c]:
				U_next[r, c] = model.R[r, c]
			U_next[r, c] = model.R[r, c] + model.gamma * np.max(np.sum(P[r, c] * U_current, axis=(1, 2)))
	return U_next

def value_iteration(model):
	'''
	Parameters:
	model - The MDP model returned by load_MDP()

	Output:
	U - The utility function, which is an M x N array
	'''
	P = compute_transition_matrix(model)
	U = np.zeros((model.M, model.N))
	while True:
		U_next = update_utility(model, P, U)
		if np.allclose(U, U_next):
			break
		U = U_next
	return U

if __name__ == "__main__":
	import utils
	model = utils.load_MDP('models/small.json')
	model.visualize()
	U = value_iteration(model)
	model.visualize(U)
