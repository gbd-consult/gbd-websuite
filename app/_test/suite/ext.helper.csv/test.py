import gws.config.loader
import gws.ext.helper.csv
import gws.types as t

import _test.util as u

root = gws.config.loader.load()

row1 = [
    t.Attribute(type=t.AttributeType.str, value='abc "q" def'),
    t.Attribute(type=t.AttributeType.str, value='12345'),
    t.Attribute(type=t.AttributeType.int, value=1234),
    t.Attribute(type=t.AttributeType.float, value=111.1234),
    t.Attribute(type=t.AttributeType.str),
]

row2 = [
    t.Attribute(type=t.AttributeType.str, value='xyz "q" uvw'),
    t.Attribute(type=t.AttributeType.str, value='23456'),
    t.Attribute(type=t.AttributeType.int, value=2345),
    t.Attribute(type=t.AttributeType.float, value=222.1234),
    t.Attribute(type=t.AttributeType.str),
]


def test_default():
    h = t.cast(gws.ext.helper.csv.Object, root.find('gws.ext.helper.csv', 'csv_default'))
    s = h.writer().write_attributes(row1).write_attributes(row2).as_str()

    exp = u.strlines('''
        "abc ""q"" def","=""12345""",1234,111.12,""
        "xyz ""q"" uvw","=""23456""",2345,222.12,""
    ''')
    assert s == exp


def test_custom():
    h = t.cast(gws.ext.helper.csv.Object, root.find('gws.ext.helper.csv', 'csv_custom'))
    s = h.writer().write_attributes(row1).write_attributes(row2).as_str()

    exp = u.strlines('''
        "abc ""q"" def";"12345";1234;111,123;""
        "xyz ""q"" uvw";"23456";2345;222,123;""
    ''')
    assert s == exp
