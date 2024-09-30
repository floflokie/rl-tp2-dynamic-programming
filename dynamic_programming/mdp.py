import random
import typing as t

import gym
from gym import spaces

"""
Partie 1 - Processus décisionnels de Markov
===========================================

Rappel: un processus décisionnel de Markov (MDP) est un modèle de
décision séquentielle dans lequel les états du système sont décrits par
une variable aléatoire, et les actions du système sont décrites par une
autre variable aléatoire. Les transitions entre états sont décrites par
une matrice de transition, et les récompenses associées à chaque
transition sont décrites par une matrice de récompenses.

Dans ce TP, nous allons utiliser la librairie `gym` pour implémenter un
MDP simple, et nous allons utiliser la programmation dynamique pour
résoudre ce MDP.
"""

# Exercice 1: MDP simple
# ----------------------
# Implémenter un MDP simple avec la librairie `gym`. Ce MDP doit avoir
# 3 états, 2 actions, et les transitions et récompenses suivantes:
#   - état 0, action 0 -> état 1, récompense -1
#   - état 0, action 1 -> état 0, récompense -1
#   - état 1, action 0 -> état 0, récompense -1
#   - état 1, action 1 -> état 2, récompense -1
#   - état 2, action 0 -> état 2, récompense 0
#   - état 2, action 1 -> état 0, récompense -1


class MDP(gym.Env):
    """
    MDP simple avec 3 états et 2 actions.
    """

    observation_space: spaces.Discrete
    action_space: spaces.Discrete

    # state, action -> [(next_state, reward, done)]
    P: list[list[tuple[int, float, bool]]]

    def __init__(self):
        # BEGIN SOLUTION
        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)
        self.P = [
            [(1, -1.0, False), (0, -1.0, False)],
            [(0, -1.0, False), (2, -1.0, False)],
            [(2, 0.0, False), (0, -1.0, False)],
        ]
        self.reset_state()
        # END SOLUTION

    def reset_state(self, value: t.Optional[int] = None):
        """
        Resets the initial state of the MDP object.

        Args:
            value (int, optional): The value to set the initial state to. If None, a random initial state is chosen.
        """
        if value is None:
            self.initial_state = random.randint(0, 2)
        else:
            self.initial_state = value

    def step(self, action: int, transition: bool = True) -> tuple[int, float, bool, dict]:  # type: ignore
        """
        Effectue une transition dans le MDP.
        Renvoie l'observation suivante, la récompense, un booléen indiquant
        si l'épisode est terminé, et un dictionnaire d'informations.
        """
        # BEGIN SOLUTION
        next_state, reward, done = self.P[self.initial_state][action]
        if transition:
            self.initial_state = next_state
        return next_state, reward, done, {}
        # END SOLUTION
