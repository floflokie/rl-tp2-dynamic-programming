# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    a = 0
    # BEGIN SOLUTION
    if n % 2 != 0:
        return 0
    dp = [0] * (n + 1)
    dp[0] = 1
    if n >= 1:
        dp[1] = 0
    if n >= 2:
        dp[2] = 3
    if n >= 3:
        dp[3] = 0
    for i in range(4, n + 1):
        dp[i] = 4 * dp[i - 2] - dp[i - 4]
    return dp[-1]
    # END SOLUTION
