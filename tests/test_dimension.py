from devito import SubsampledDimension, Grid, SteppingDimension # noqa


def test_subsampled_dimension():
    grid = Grid(shape=(11, 11))
    time = grid.time_dim # noqa
    time_subsampled = SteppingDimension('t_sub', parent=time, modulo=4) # noqa
