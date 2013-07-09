import os

import pytest

from .extractor_benchmarks import benchmark_extractors, benchmark_file, hashes


@pytest.mark.parametrize(['ex_cls'], [(b,) for b in benchmark_extractors])
def test_benchmark(ex_cls):
    path = os.path.join(os.path.dirname(__file__), benchmark_file(ex_cls))
    expected = open(path).read()
    result = hashes(ex_cls)
    assert result == expected
