"""2D initial conditions: hot (noise), log-normal hot, perfect crystal,
crystal with vacancies, dislocation ring.

STUB. Migration note — the audit found that `seed=None` and the
`np.random.seed(seed)` call are present in eight of nine forks and missing from
the single-precision fork, making its runs unreproducible. The migrated version
takes an explicit `rng` (a `numpy.random.Generator`), not a global seed:

    def hot(geometry, mean_density, std, k_max, *, rng): ...

Passing a Generator rather than seeding the global state is what makes a run
reproducible under parallelism.
"""

raise NotImplementedError("fields.initial_conditions_2d not migrated yet")
