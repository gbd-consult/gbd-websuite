import gws.lib.units

import _test.util as u


def test_parse():
    nn, uu = gws.lib.units.parse('24.5mm', units=['px', 'mm'])
    assert (nn, uu) == (24.5, 'mm')

    nn, uu = gws.lib.units.parse('24.5 m', units=['px', 'mm'])
    assert (nn, uu) == (24500, 'mm')

    nn, uu = gws.lib.units.parse('1234 mm', units=['px', 'm'])
    assert (nn, uu) == (1.234, 'm')

    nn, uu = gws.lib.units.parse('1234 cm', units=['px', 'm'])
    assert (nn, uu) == (12.34, 'm')

    nn, uu = gws.lib.units.parse('1234 cm', units=['px', 'km'])
    assert (nn, uu) == (0.01234, 'km')

    nn, uu = gws.lib.units.parse(1234, units=['px', 'm'], default='px')
    assert (nn, uu) == (1234, 'px')

    nn, uu = gws.lib.units.parse('1234', units=['px', 'm'], default='px')
    assert (nn, uu) == (1234, 'px')

    with u.raises(ValueError):
        nn, uu = gws.lib.units.parse('1234', units=['px', 'm'])

    with u.raises(ValueError):
        nn, uu = gws.lib.units.parse('1234 in', units=['px', 'm'])

    with u.raises(ValueError):
        nn, uu = gws.lib.units.parse('1234 BLAH', units=['px', 'm'])
