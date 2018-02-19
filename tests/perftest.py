from functools import reduce
from operator import mul

import numpy as np
from sympy import solve

from devito import Eq, Grid, Operator, TimeFunction, configuration


def operator(shape, time_order, **kwargs):
    grid = Grid(shape=shape)
    spacing = 0.1
    a = 0.5
    c = 0.5
    dx2, dy2 = spacing**2, spacing**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))

    # Allocate the grid and set initial condition
    # Note: This should be made simpler through the use of defaults
    u = TimeFunction(name='u', grid=grid, time_order=time_order, space_order=2)
    u.data[0, :] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)

    # Derive the stencil according to devito conventions
    eqn = Eq(u.dt, a * (u.dx2 + u.dy2) - c * (u.dxl + u.dyl))
    stencil = solve(eqn, u.forward, rational=False)[0]
    op = Operator(Eq(u.forward, stencil), **kwargs)

    # Execute the generated Devito stencil operator
    op.apply(u=u, t=30, dt=dt)
    return u.data[1, :], op


def no_blocking(shape):
    configuration['skew_factor'] = 0
    return operator(shape, time_order=2, dle='noop')


def space_blocking(shape, blockshape):
    configuration['skew_factor'] = 0
    return operator(shape, time_order=2, dse='skewing',
                    dle=('blocking,openmp', {'blockshape': blockshape,
                                             'blockinner': False}))

def time_blocking(shape, blockshape):
    configuration['skew_factor'] = 2
    return operator(shape, time_order=2, dse='skewing',
                    dle=('blocking,openmp', {'blockshape': blockshape,
                                             'blockinner': False}))


if __name__ == '__main__':
    shape = (400, 400, 400)
    blockshape = (16, 16, 16)

    print("Running time-tiling code")
    tm_bl, _ = time_blocking(shape, blockshape)

    print("Running code without tiling")
    no_bl, _ = no_blocking(shape)

    print("Running space-tiling code")
    sp_bl, _ = space_blocking(shape, blockshape)


    assert np.equal(no_bl.data, tm_bl.data).all()
    assert np.equal(no_bl.data, sp_bl.data).all()

    print("Ran successfully")
