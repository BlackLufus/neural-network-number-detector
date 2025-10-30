import numpy as np

permutation = np.random.permutation(range(10))

print(permutation)

x = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
print("Shuffled x:", x[permutation])