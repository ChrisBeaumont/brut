from ..field import get_field


def test_get_field():
    f1 = get_field(30)
    assert f1.lon == 30


def test_field_cached():
    f1 = get_field(30)
    f2 = get_field(30)
    assert f1 is f2
    f3 = get_field(31)
    assert f3 is not f2
    f3 = get_field(30)
    assert f3 is not f2
