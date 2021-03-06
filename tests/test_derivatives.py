import numpy as np
import pytest
from conftest import t, x, y, z, skipif_yask
from sympy import Derivative, simplify

from devito import Grid, Function, TimeFunction


@pytest.fixture
def shape(xdim=20, ydim=30):
    return (xdim, ydim)


@pytest.fixture
def grid(shape):
    return Grid(shape=shape)


@skipif_yask
@pytest.mark.parametrize('SymbolType, dimension', [
    (Function, x), (Function, y),
    (TimeFunction, x), (TimeFunction, y), (TimeFunction, t),
])
def test_stencil_derivative(grid, shape, SymbolType, dimension):
    """Test symbolic behaviour when expanding stencil derivatives"""
    u = SymbolType(name='u', grid=grid)
    u.data[:] = 66.6
    dx = u.diff(x)
    dxx = u.diff(x, x)
    # Check for sympy Derivative objects
    assert(isinstance(dx, Derivative) and isinstance(dxx, Derivative))
    s_dx = dx.as_finite_difference([x - x.spacing, x])
    s_dxx = dxx.as_finite_difference([x - x.spacing, x, x + x.spacing])
    # Check stencil length of first and second derivatives
    assert(len(s_dx.args) == 2 and len(s_dxx.args) == 3)
    u_dx = s_dx.args[0].args[1]
    u_dxx = s_dx.args[0].args[1]
    # Ensure that devito meta-data survived symbolic transformation
    assert(u_dx.shape[-2:] == shape and u_dxx.shape[-2:] == shape)
    assert(np.allclose(u_dx.data, 66.6))
    assert(np.allclose(u_dxx.data, 66.6))


@skipif_yask
@pytest.mark.parametrize('SymbolType, derivative, dim', [
    (Function, 'dx2', 3), (Function, 'dy2', 3),
    (TimeFunction, 'dx2', 3), (TimeFunction, 'dy2', 3), (TimeFunction, 'dt', 2)
])
def test_preformed_derivatives(grid, SymbolType, derivative, dim):
    """Test the stencil expressions provided by devito objects"""
    u = SymbolType(name='u', grid=grid, time_order=2, space_order=2)
    expr = getattr(u, derivative)
    assert(len(expr.args) == dim)


@skipif_yask
@pytest.mark.parametrize('derivative, dimension', [
    ('dx', x), ('dy', y), ('dz', z)
])
@pytest.mark.parametrize('order', [1, 2, 4, 6, 8, 10, 12, 14, 16])
def test_derivatives_space(derivative, dimension, order):
    """Test first derivative expressions against native sympy"""
    grid = Grid(shape=(20, 20, 20))
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=order)
    expr = getattr(u, derivative)
    # Establish native sympy derivative expression
    width = int(order / 2)
    if order == 1:
        indices = [dimension, dimension + dimension.spacing]
    else:
        indices = [(dimension + i * dimension.spacing)
                   for i in range(-width, width + 1)]
    s_expr = u.diff(dimension).as_finite_difference(indices)
    assert(simplify(expr - s_expr) == 0)  # Symbolic equality
    assert(expr == s_expr)  # Exact equailty


@skipif_yask
@pytest.mark.parametrize('derivative, dimension', [
    ('dx2', x), ('dy2', y), ('dz2', z)
])
@pytest.mark.parametrize('order', [2, 4, 6, 8, 10, 12, 14, 16])
def test_second_derivatives_space(derivative, dimension, order):
    """Test second derivative expressions against native sympy"""
    grid = Grid(shape=(20, 20, 20))
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=order)
    expr = getattr(u, derivative)
    # Establish native sympy derivative expression
    width = int(order / 2)
    indices = [(dimension + i * dimension.spacing)
               for i in range(-width, width + 1)]
    s_expr = u.diff(dimension, dimension).as_finite_difference(indices)
    assert(simplify(expr - s_expr) == 0)  # Symbolic equality
    assert(expr == s_expr)  # Exact equailty
