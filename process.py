import numpy as np
import polars as pl

from matplotlib import pyplot as plt
from matplotlib import use as pltuse

pltuse('module://matplotlib-backend-sixel')

L = [16, 32, 64, 128, 256]
# cutoffs = [110_000, 2_000, 100_000, 100_000]

for length in L:
    file = f'./data/energy/L{length}-j1.csv'

    df = pl.read_csv(file, has_header=True)
    betas = np.array(list(map(float, df.columns)))

    energies = np.asarray(df)[-6_000 * int(length):] / length

    mean_energy = energies.mean(axis=0, keepdims=True)
    fluctuations = energies**2 - mean_energy**2

    heat_capacity = betas**2 * fluctuations.mean(axis=0)

    plt.plot(1 / betas, heat_capacity, label=f'{length}')

plt.axvline(1.043)
plt.xlabel(r'$T$')
plt.ylabel(r'$C_V / L^2 = \beta^2 (\langle E^2 \rangle - \langle E \rangle^2) / L^2$')
plt.legend()
plt.savefig('./plots/heat_capacity_metropolis.png')
