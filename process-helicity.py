import numpy as np
import polars as pl
import os

from matplotlib import pyplot as plt
from matplotlib import use as pltuse

pltuse('module://matplotlib-backend-sixel')

L = [4, 8, 16, 32, 64, 128, 256]
# cutoffs = [110_000, 2_000, 100_000, 100_000]

for length in L:
    file = f'./data/helicity/L{length}-j1.csv'

    if not os.path.exists(file):
        continue

    df = pl.read_csv(file, has_header=True)
    betas = np.array(list(map(float, df.columns)))

    helicities = np.asarray(df)[-90_000:]

    plt.plot(1 / betas, helicities.mean(axis=0), label=f'{length}')

plt.axvline(0.88)
plt.xlabel(r'$T$')
plt.ylabel(r'$\Upsilon$')
plt.legend()
# plt.show()
plt.savefig('./plots/helicity_metropolis.png')
