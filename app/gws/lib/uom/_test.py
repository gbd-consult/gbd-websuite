import gws
import gws.lib.uom as uom
import gws.test.util as u
import random


def test_scale_to_res():
    assert uom.scale_to_res(-15151) == -4.24228
    assert uom.scale_to_res(-0.3) == -8.4e-05
    assert uom.scale_to_res(0) == 0
    assert uom.scale_to_res(0.7) == 0.00019599999999999997
    assert uom.scale_to_res(98725167) == 27643.046759999997

def test_res_to_scale():
    assert uom.res_to_scale(-4.24228) == -15151
    assert uom.res_to_scale(-8.4e-05) == int(-0.3)
    assert uom.res_to_scale(0) == 0
    assert uom.res_to_scale(0.001) == 3
    assert uom.res_to_scale(27643.046759999997) == 98725167

def test_mm_to_px():
    """
    This tests uses snapshotted values and detects change (most likely precision?).
    It is not meant to catch extremes or verify a defined spec.
    """
    tests = [
        ((805052, 42718), 1353945328.1889765), 
        ((485568, -7375), -140986771.65354332), 
        ((357767, 6842), 96371724.96062993), 
        ((-737414, 68376), -1985095262.3622048), 
        ((-280698, 37266), -411830380.62992126), 
        ((981625, 83293), 3218995713.5826774), 
        ((381642, 57759), 867844892.8346457), 
        ((-540230, -2102), 44707222.83464567), 
        ((-343019, 66888), -903301372.9133859), 
        ((495243, 28853), 562568751.1417323)
    ]

    for test in tests:
        params, result = test
        number, ppi = params

        assert uom.mm_to_px(number, ppi) == result

def test_to_px():
    """
        Test general conversion of Measurements to pixels

        In it's current state we test the ability to convert
        all supported units, not the correctness of these conversions.

        #TODO: Create Tests for correctnes of conversions according to 
        # whatever standards there are for a given unit.
    """
    obj_uom= gws.core.types.Uom()
    # TODO use this line to test aginst all units, but all units are not implemented yet
    # units = [attr for attr in dir(obj_uom) if not callable(getattr(obj_uom, attr)) and not attr.startswith("_")]
    units = [ 'MM', 'PX' ]
    for unit in units:
        mm = (1.1, getattr(obj_uom, unit))
        ppi = 100
        assert uom.to_px(mm, ppi)


def test_size_to_px():
    """
        Assuming test_mm_to_px() passes, this should be sufficient
    """
    assert uom.size_mm_to_px((2345,2345), 123) == (11355.708661417324, 11355.708661417324)


def test_msize_to_px():
    syu = (1,2, 'mm')
    ppi = 123
    asd = uom.msize_to_px(syu, ppi)
    assert asd == (4.84251968503937, 9.68503937007874, 'px')


def test_px_to_mm():
    res = uom.px_to_mm(1234, 123)
    assert res == 254.8260162601626


def test_to_mm():
    # TODO similar to test_to_px we could test for all possible units here once their implemented
    assert uom.to_mm((100, 'mm'), 123) == (100, 'mm')
    assert uom.to_mm((100, 'px'), 123) == (20.65040650406504, 'mm')


def test_size_px_to_mm():
    assert uom.size_px_to_mm((1234,1234), 123) == (254.8260162601626,254.8260162601626)


def test_msize_to_mm():
    xyu = 1,2, 'px'
    ppi = 123
    res = uom.msize_to_mm(xyu, ppi)
    assert res == (0.20650406504065041, 0.41300813008130083, 'mm')


def test_parse():
    inputs = [
        ('1 m',       (1.0, 'm')),
        ('10 m',      (10.0, 'm')),
        ('-10 m',     (-10.0, 'm')),
        #('1,2 m',     (1.2, 'm')),
        ('1.245 m',   (1.245, 'm')),
        #('0.21 us-in',(0.21, 'us-in'))
    ]
    for test, res in inputs:
        assert uom.parse(test) == res



def test_parse_duration():
    assert uom.parse_duration(123) == 123
    assert uom.parse_duration('123') == 123