import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    converge = False
    i = 0
    # tant que la valeur de chaque état n'a pas convergé et qu'on n'a pas atteint le max_iter
    while(not converge and i < max_iter):
        delta = 0.0
        # pour chaque état estimer la fonction de valeur
        for state in range(mdp.observation_space.n):
            # on garde en copy la valeur courante de cet état
            v = values[state].copy()
            max_a = -np.inf
            # choisir l'action qui maximise la valeur
            for action in range(mdp.action_space.n):
                next_state, reward, done = mdp.P[state][action]
                action_value = reward + gamma * values[next_state]
                max_a = max(max_a, action_value)
            # mettre à jour la valeur de l'état
            values[state] = max_a
            delta = max(delta, abs(v - values[state]))

        if delta < 1e-5:
            converge = True
        i += 1

    # END SOLUTION
    return values


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))
    print(env.grid)
    # BEGIN SOLUTION
    converge = False
    i = 0
    # tant que la valeur de chaque état n'a pas convergé et qu'on n'a pas atteint le max_iter
    while(not converge and i < max_iter):
        delta = 0.0
        prev_values = values.copy()
        # pour chaque case estimer la fonction de valeur
        for row in range(env.height):
            for col in range(env.width):
                
                # Si la case est un état terminal, on ne fait rien
                if env.grid[row, col] == 'P' or env.grid[row, col] == 'N':
                    continue
                # Si la case est un mur, on met la valeur à 0 et on s'arrête
                if env.grid[row, col] == 'W':
                    values[row, col] = 0
                    continue
                
                env.set_state(row, col)
                max_value = -np.inf
                
                # Choisir l'action qui maximise la valeur de l'état
                for action in range(env.action_space.n):
                    next_state, reward, _, _ = env.step(action, make_move=False)
                    next_row, next_col = next_state
                    value = reward + gamma * prev_values[next_row, next_col]
                    max_value = max(max_value, value)
                
                # Mettre à jour la valeur de l'état
                values[row, col] = max_value
                delta = max(delta, abs(values[row, col] - prev_values[row, col]))
        
        if delta < theta:
            converge = True
        i += 1
    
    return values
    # END SOLUTION


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            # print((row, col), next_state, reward, probability)
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    diff = float("inf")
    iteration_count = 0

    while iteration_count < max_iter:
        temp_values = values.copy()

        for x in range(4):
            for y in range(4):
                env.set_state(x, y)
                diff = value_iteration_per_state(env, values, gamma, temp_values, diff)
        
        if diff < theta:
            break
        
        iteration_count += 1
    return values
    # END SOLUTION