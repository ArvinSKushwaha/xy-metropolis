from dataclasses import dataclass
from functools import partial

import jax

from jax import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray, Scalar


@dataclass
class XYModel:
    n: int
    j: float
    spins: Float[Array, '{self.n} {self.n}']
    energy: Float[Scalar, '']

    @staticmethod
    def initialize_state(n: int, j: float, spins: Float[Array, '{n} {n}']) -> 'XYModel':
        xy = XYModel(n, j, spins, np.zeros(()))
        return XYModel(n, j, spins, xy.total_energy())

    @jax.jit
    def energy_difference(
        self, i: int, j: int, rotation: Float[Scalar, '']
    ) -> Float[Scalar, '']:
        N = self.n
        spins = (
            np.array(
                [
                    self.spins[(i - 1) % N, j],
                    self.spins[(i + 1) % N, j],
                    self.spins[i, (j - 1) % N],
                    self.spins[i, (j + 1) % N],
                ]
            )
            - self.spins[i, j]
        )
        old_energy = -self.j * np.cos(spins).sum()
        new_energy = -self.j * np.cos(spins - rotation).sum()
        return new_energy - old_energy

    @jax.jit
    def total_energy(
        self,
    ) -> Float[Scalar, '']:
        return -self.j * (
            np.cos(np.roll(self.spins, 1, 0) - self.spins).sum()
            + np.cos(np.roll(self.spins, 1, 1) - self.spins).sum()
        )

    @staticmethod
    @jax.jit
    def helicity_modulus(
        state: 'XYModel', beta: Float[Scalar, '']
    ) -> Float[Scalar, '']:
        e = (1 / state.n**2) * (
            np.cos(np.roll(state.spins, 1, 0) - state.spins).sum()
            + np.cos(np.roll(state.spins, 1, 1) - state.spins).sum()
        )
        s = (1 / state.n**2) * (
            np.sin(np.roll(state.spins, 1, 0) - state.spins).sum()
            + np.sin(np.roll(state.spins, 1, 1) - state.spins).sum()
        )

        return e - state.n**2 * beta * s**2


jax.tree_util.register_dataclass(
    XYModel,
    data_fields=['spins', 'energy'],
    meta_fields=['n', 'j'],
)


@partial(jax.jit, static_argnames=['iters'])
def metropolis_update(
    state: XYModel,
    beta: float,
    key: PRNGKeyArray,
    iters: int,
) -> tuple[XYModel, PRNGKeyArray]:
    N = state.n

    def update(_: int, a: tuple[XYModel, PRNGKeyArray]) -> tuple[XYModel, PRNGKeyArray]:
        state, key = a
        for n in range(4):
            key, update_key, sample_key = jax.random.split(key, 3)
            i, j = np.indices((N // 2, N // 2)) * 2
            i: Array = i.ravel() + (n // 2)
            j: Array = j.ravel() + (n % 2)
            rotations = jax.random.uniform(
                update_key,
                shape=(N // 2, N // 2),
                minval=-np.pi / 4,
                maxval=np.pi / 4,
            )

            delta_energy = jax.vmap(state.energy_difference)(
                i,  # type: ignore
                j,  # type: ignore
                rotations.ravel(),
            ).reshape((N // 2, N // 2))
            probabilities = jax.random.uniform(sample_key, shape=(N // 2, N // 2))

            spins = state.spins.at[n // 2 :: 2, n % 2 :: 2].add(
                np.where(probabilities < np.exp(-beta * delta_energy), rotations, 0.0)
            )
            total_delta_energy = np.where(
                probabilities < np.exp(-beta * delta_energy),
                delta_energy,
                0.0,
            ).sum()
            state = XYModel(N, state.j, spins, state.energy + total_delta_energy)
        return state, key

    return jax.lax.fori_loop(0, iters, update, (state, key))


key = jax.random.key(0)
key, spin_key = jax.random.split(key)

betas = np.linspace(0.666, 1.5, 100)

J = 1
Ns = [64, 128, 256]

for N in Ns:
    xy = jax.vmap(XYModel.initialize_state, (None, None, 0))(
        N,
        J,
        np.zeros((betas.size, N, N)),
    )

    keys = jax.random.split(key, betas.size)

    with open(f'./data/energy/L{N}-j{J}.csv', 'w') as f:
        f.write(','.join(map(str, betas.tolist())))

    with open(f'./data/helicity/L{N}-j{J}.csv', 'w') as f:
        f.write(','.join(map(str, betas.tolist())))

    metropolis_updates = jax.jit(jax.vmap(metropolis_update))

    with (
        open(f'./data/helicity/L{N}-j{J}.csv', 'a') as f_helicity,
        open(f'./data/energy/L{N}-j{J}.csv', 'a') as f_energy,
    ):
        for i in range(100_000):
            xy, keys = metropolis_updates(xy, betas, keys, 10)

            helicity = jax.vmap(XYModel.helicity_modulus)(xy, betas)
            energy = xy.energies

            f_helicity.write('\n' + ','.join(map(str, helicity.tolist())))
            f_energy.write('\n' + ','.join(map(str, energy.tolist())))

            if i % 100 == 0:
                f_helicity.flush()
                f_energy.flush()
