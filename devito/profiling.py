from __future__ import absolute_import

import operator
from collections import OrderedDict, namedtuple
from functools import reduce

from ctypes import Structure, byref, c_double
from cgen import Struct, Value

from devito.ir.iet import Expression, TimedList, FindSections, FindNodes, Transformer
from devito.symbolics import estimate_cost, estimate_memory

__all__ = ['Profile', 'create_profile']


def create_profile(node):
    """
    Create a :class:`Profiler` for the Iteration/Expression tree ``node``.
    The following code sections are profiled: ::

        * The whole ``node``;
        * A sequence of perfectly nested loops that have common :class:`Iteration`
          dimensions, but possibly different extent. For example: ::

            for x = 0 to N
              ..
            for x = 1 to N-1
              ..

          Both Iterations have dimension ``x``, and will be profiled as a single
          section, though their extent is different.
        * Any perfectly nested loops.
    """
    profiler = Profiler()

    # Group by root Iteration
    mapper = OrderedDict()
    for itspace in FindSections().visit(node):
        mapper.setdefault(itspace[0], []).append(itspace)

    # Group sections if their iteration spaces overlap
    key = lambda itspace: set([i.dim for i in itspace])
    found = []
    for v in mapper.values():
        queue = list(v)
        handle = []
        while queue:
            item = queue.pop(0)
            if not handle or key(item) == key(handle[0]):
                handle.append(item)
            else:
                # Found a timing section
                found.append(tuple(handle))
                handle = [item]
        if handle:
            found.append(tuple(handle))

    # Create and track C-level timers
    mapper = OrderedDict()
    for i, group in enumerate(found):
        name = 'section_%d' % i

        # We time at the single timestep level
        for i in zip(*group):
            root = i[0]
            remainder = tuple(j for j in i if j is not root)
            if not (root.dim.is_Time or root.dim.is_Stepping):
                break

        # Prepare to transform the Iteration/Expression tree
        body = (root,) + remainder
        mapper[root] = TimedList(gname=profiler.varname, lname=name, body=body)
        mapper.update(OrderedDict([(j, None) for j in remainder]))

        # Estimate computational properties of the profiled section
        expressions = FindNodes(Expression).visit(body)
        ops = estimate_cost([e.expr for e in expressions])
        memory = estimate_memory([e.expr for e in expressions])

        # Keep track of the new profiled section
        profiler.add(name, group[0], ops, memory)

    # Transform the Iteration/Expression tree introducing the C-level timers
    processed = Transformer(mapper).visit(node)

    return processed, profiler


class Profiler(object):

    """
    A Profiler is used to manage profiling information for Devito generated C code.
    """

    varname = "timings"
    structname = "profile"

    def __init__(self):
        # To be populated as new sections are tracked
        self._sections = OrderedDict()
        self._C_timings = None

    def add(self, name, section, ops, memory):
        """
        Add a profiling section.

        :param name: The name which uniquely identifies the profiled code section.
        :param section: The code section, represented as a tuple of :class:`Iteration`s.
        :param ops: The number of floating-point operations in the section.
        :param memory: The memory traffic in the section, as bytes moved from/to memory.
        """
        self._sections[section] = Profile(name, ops, memory)

    def setup(self):
        """
        Allocate and return a pointer to the timers C-level Struct, which includes
        all timers added to ``self`` through ``self.add(...)``.
        """
        self._C_timings = self.dtype()
        return byref(self._C_timings)

    def summary(self, arguments, dtype):
        """
        Return a summary of the performance numbers measured.

        :param dim_sizes: The run-time extent of each :class:`Iteration` tracked
                          by this Profiler. Used to compute the operational intensity
                          and the perfomance achieved in GFlops/s.
        :param dtype: The data type of the objects in the profiled sections. Used
                      to compute the operational intensity.
        """

        summary = PerformanceSummary()
        for itspace, profile in self._sections.items():
            dims = {i: i.dim.parent if i.dim.is_Stepping else i.dim for i in itspace}

            # Time
            time = self.timings[profile.name]

            # Flops
            itershape = [i.extent(finish=arguments[dims[i].end_name],
                                  start=arguments[dims[i].start_name]) for i in itspace]
            iterspace = reduce(operator.mul, itershape)
            flops = float(profile.ops*iterspace)
            gflops = flops/10**9
            gpoints = iterspace/10**9

            # Compulsory traffic
            datashape = [(arguments[dims[i].end_name] - arguments[dims[i].start_name])
                         for i in itspace]
            dataspace = reduce(operator.mul, datashape)
            traffic = float(profile.memory*dataspace*dtype().itemsize)
            # Derived metrics
            oi = flops/traffic
            gflopss = gflops/time
            gpointss = gpoints/time

            # Keep track of performance achieved
            summary.setsection(profile.name, time, gflopss, gpointss, oi, profile.ops,
                               itershape, datashape)

        # Rename the most time consuming section as 'main'
        if len(summary) > 0:
            summary['main'] = summary.pop(max(summary, key=summary.get))

        return summary

    @property
    def timings(self):
        """
        Return the timings, up to microseconds, as a dictionary.
        """
        if self._C_timings is None:
            raise RuntimeError("Cannot extract timings with non-finalized Profiler.")
        return {field: max(getattr(self._C_timings, field), 10**-6)
                for field, _ in self._C_timings._fields_}

    @property
    def dtype(self):
        """
        Return the profiler C type in ctypes format.
        """
        return type(Profiler.structname, (Structure,),
                    {"_fields_": [(i.name, c_double) for i in self._sections.values()]})

    @property
    def cdef(self):
        """
        Returns a :class:`cgen.Struct` representing the profiler data structure in C
        (a ``struct``).
        """
        return Struct(Profiler.structname,
                      [Value('double', i.name) for i in self._sections.values()])


class PerformanceSummary(OrderedDict):

    """
    A special dictionary to track and quickly access performance data.
    """

    def setsection(self, key, time, gflopss, gpointss, oi, ops, itershape, datashape):
        self[key] = PerfEntry(time, gflopss, gpointss, oi, ops, itershape, datashape)

    @property
    def gflopss(self):
        return OrderedDict([(k, v.gflopss) for k, v in self.items()])

    @property
    def oi(self):
        return OrderedDict([(k, v.oi) for k, v in self.items()])

    @property
    def timings(self):
        return OrderedDict([(k, v.time) for k, v in self.items()])


Profile = namedtuple('Profile', 'name ops memory')
"""Metadata for a profiled code section."""


PerfEntry = namedtuple('PerfEntry', 'time gflopss gpointss oi ops itershape datashape')
"""Structured performance data."""
