import os

import pytest
from mock import MagicMock

from .extractor_benchmarks import benchmark_extractors, benchmark_file, hashes
from ..extractors import RGBExtractor

@pytest.mark.parametrize(['ex_cls'], [(b,) for b in benchmark_extractors])
def test_benchmark(ex_cls):
    path = os.path.join(os.path.dirname(__file__), benchmark_file(ex_cls))
    expected = open(path).read()
    result = hashes(ex_cls)
    assert result == expected

def test_preprocessor():

    ex = RGBExtractor()
    p = MagicMock()

    ex.preprocessors.append(p)

    ex.extract(*(309, 309.059036, 0.1660606, 0.12063246))
    assert p.call_count == 1
