from hashlib import md5

from bubbly import extractors as ex
from bubbly.dr1 import on_stamp_params

benchmark_params = list(on_stamp_params())[::100]
benchmark_extractors = [ex.RGBExtractor,
                        ex.EllipseExtractor,
                        ex.RingWaveletCompressionStatExtractor,
                        ex.RawStatsExtractor,
                        ex.CompressionExtractor]


def hashes(ex_cls):
    ex = ex_cls()
    result = [md5(ex.extract(*p)).hexdigest() for p in benchmark_params]
    return '\n'.join(result)


def benchmark_file(ex_cls):
    return "{0}.md5".format(ex_cls.__name__)


def _build(ex_cls, path):
    with open(path, 'w') as outfile:
        outfile.write(hashes(ex_cls))


def build_all_benchmarks():
    for ex in benchmark_extractors:
        _build(ex, benchmark_file(ex))


if __name__ == "__main__":
    build_all_benchmarks()
