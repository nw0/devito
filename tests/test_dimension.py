from devito import SubsampledDimension, Grid, SteppingDimension


def test_subsampled_dimension():
    grid = Grid(shape=(11, 11))
    time = grid.time_dim
    time_subsampled = SteppingDimension('t_sub', parent=time, modulo=4)
    
