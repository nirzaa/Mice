import numpy as np

Ts_x = [round(T,2) for T in np.linspace(0.1, 4, 40)]
Ts_y = [0.1, 1, 2, 2.1, 2.2, 2.3, 2.4, 2.7, 2.9, 3, 3.2, 4]
Ts = [i for i in Ts_x if i not in Ts_y]
print(Ts_x)
print(Ts_y)
print(Ts)