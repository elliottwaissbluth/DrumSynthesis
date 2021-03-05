from pathlib import Path
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from sklearn.preprocessing import MinMaxScaler
import cvxpy as cp

path = Path.cwd()
samples_tensor = np.load(path / 'Drum Synthesis' / 'samples_tensor.npy')

# Formulate SOCP problem
n = samples_tensor.T.shape[1]
X = samples_tensor.T.astype(np.double)

kappa = cp.Variable()
kappas = [cp.Variable() for _ in range(n)]
W = cp.Variable((n, n))

soc_constraints = []
soc_constraints.append(kappa == 1)
soc_constraints.append(sum(kappas) <= kappa)
soc_constraints.extend([cp.SOC(kappas[i], W.T[i]) for i in range(n)])

obj = cp.Minimize(cp.norm(X - X @ W.T, 'fro'))
prob = cp.Problem(obj, soc_constraints)

prob.solve()

try:
    print('Hello I\'m here')
    weights = np.array(W.value)
    np.save('exemplar_weights.npy', weights)
except:
    print('Could not save weights')
    print(W.value)