"""Tests for the element module"""

import pytest
import gws

import gws.lib.xmlx.element


def _make(*args, **kwargs):
    return gws.lib.xmlx.element.XmlElement(*args, **kwargs)


def test_init():
    # Test basic initialization
    e = _make('test')
    assert e.tag == 'test'
    assert e.name == 'test'
    assert e.lcName == 'test'
    assert e.text == ''
    assert e.tail == ''
    assert e.attrib == {}
    assert len(e) == 0

    # Test with attributes
    e = _make('test', {'attr1': 'value1'}, attr2='value2')
    assert e.attrib == {'attr1': 'value1', 'attr2': 'value2'}


def test_repr():
    e = _make('test')
    repr_str = repr(e)
    assert 'XmlElement' in repr_str
    assert "'test'" in repr_str


def test_makeelement():
    e = _make('parent')
    child = e.makeelement('child', {'attr': 'value'})
    assert isinstance(child, e.__class__)
    assert child.tag == 'child'
    assert child.attrib == {'attr': 'value'}


def test_copy():
    e = _make('test', {'attr': 'value'})
    e.text = 'hello'
    e.tail = 'world'
    child = _make('child')
    e.append(child)

    copied = e.__copy__()
    assert copied.tag == e.tag
    assert copied.attrib == e.attrib
    assert copied.text == e.text
    assert copied.tail == e.tail
    assert len(copied) == len(e)
    assert copied[0].tag == 'child'


def test_len():
    e = _make('parent')
    assert len(e) == 0

    e.append(_make('child1'))
    assert len(e) == 1

    e.append(_make('child2'))
    assert len(e) == 2


def test_getitem_setitem_delitem():
    parent = _make('parent')
    child1 = _make('child1')
    child2 = _make('child2')
    child3 = _make('child3')

    parent.append(child1)
    parent.append(child2)

    # Test getitem
    assert parent[0] is child1
    assert parent[1] is child2

    # Test setitem
    parent[1] = child3
    assert parent[1] is child3

    # Test delitem
    del parent[0]
    assert len(parent) == 1
    assert parent[0] is child3


def test_append():
    parent = _make('parent')
    child = _make('child')

    parent.append(child)
    assert len(parent) == 1
    assert parent[0] is child


def test_extend():
    parent = _make('parent')
    child1 = _make('child1')
    child2 = _make('child2')

    parent.extend([child1, child2])
    assert len(parent) == 2
    assert parent[0] is child1
    assert parent[1] is child2


def test_insert():
    parent = _make('parent')
    child1 = _make('child1')
    child2 = _make('child2')
    child3 = _make('child3')

    parent.append(child1)
    parent.append(child2)
    parent.insert(1, child3)

    assert len(parent) == 3
    assert parent[0] is child1
    assert parent[1] is child3
    assert parent[2] is child2


def test_remove():
    parent = _make('parent')
    child1 = _make('child1')
    child2 = _make('child2')

    parent.append(child1)
    parent.append(child2)
    parent.remove(child1)

    assert len(parent) == 1
    assert parent[0] is child2


def test_find():
    parent = _make('parent')
    child1 = _make('child')
    child2 = _make('other')

    parent.append(child1)
    parent.append(child2)

    found = parent.find('child')
    assert found is child1

    not_found = parent.find('nonexistent')
    assert not_found is None


def test_findtext():
    parent = _make('parent')
    child = _make('child')
    child.text = 'test text'
    parent.append(child)

    text = parent.findtext('child')
    assert text == 'test text'

    default_text = parent.findtext('nonexistent', 'default')
    assert default_text == 'default'


def test_findall():
    parent = _make('parent')
    child1 = _make('child')
    child2 = _make('child')
    other = _make('other')

    parent.append(child1)
    parent.append(child2)
    parent.append(other)

    children = parent.findall('child')
    assert len(children) == 2
    assert child1 in children
    assert child2 in children


def test_iterfind():
    parent = _make('parent')
    child1 = _make('child')
    child2 = _make('child')

    parent.append(child1)
    parent.append(child2)

    found_children = list(parent.iterfind('child'))
    assert len(found_children) == 2
    assert child1 in found_children
    assert child2 in found_children


def test_clear():
    e = _make('test', {'attr': 'value'})
    e.text = 'hello'
    e.tail = 'world'
    e.append(_make('child'))

    e.clear()
    assert e.attrib == {}
    assert e.text == ''
    assert e.tail == ''
    assert len(e) == 0


def test_get_set():
    e = _make('test')

    # Test get with default
    assert e.get('nonexistent') is None
    assert e.get('nonexistent', 'default') == 'default'

    # Test set and get
    e.set('attr', 'value')
    assert e.get('attr') == 'value'


def test_keys():
    e = _make('test', {'attr1': 'value1', 'attr2': 'value2'})
    keys = list(e.keys())
    assert 'attr1' in keys
    assert 'attr2' in keys
    assert len(keys) == 2


def test_items():
    e = _make('test', {'attr1': 'value1', 'attr2': 'value2'})
    items = list(e.items())
    assert ('attr1', 'value1') in items
    assert ('attr2', 'value2') in items
    assert len(items) == 2


def test_iter():
    parent = _make('parent')
    child1 = _make('child')
    child2 = _make('other')
    grandchild = _make('child')
    child1.append(grandchild)

    parent.append(child1)
    parent.append(child2)

    # Test iter all
    all_elements = list(parent.iter())
    assert len(all_elements) == 4  # parent, child1, grandchild, child2
    assert parent in all_elements
    assert child1 in all_elements
    assert grandchild in all_elements
    assert child2 in all_elements

    # Test iter with tag
    child_elements = list(parent.iter('child'))
    assert len(child_elements) == 2
    assert child1 in child_elements
    assert grandchild in child_elements

    # Test iter with '*'
    all_with_star = list(parent.iter('*'))
    assert len(all_with_star) == 4


def test_itertext():
    parent = _make('parent')
    parent.text = 'parent text'

    child = _make('child')
    child.text = 'child text'
    child.tail = 'child tail'

    parent.append(child)

    texts = list(parent.itertext())
    assert 'parent text' in texts
    assert 'child text' in texts
    assert 'child tail' in texts


def test_bool():
    e = _make('test')
    assert bool(e) is True


def test_iter_children():
    parent = _make('parent')
    child1 = _make('child1')
    child2 = _make('child2')

    parent.append(child1)
    parent.append(child2)

    children = list(parent)
    assert len(children) == 2
    assert child1 in children
    assert child2 in children


def test_children():
    parent = _make('parent')
    child1 = _make('child1')
    child2 = _make('child2')

    parent.append(child1)
    parent.append(child2)

    children = parent.children()
    assert len(children) == 2
    assert child1 in children
    assert child2 in children


def test_has():
    e = _make('test', {'attr1': 'value1'})
    assert e.has('attr1') is True
    assert e.has('nonexistent') is False


def test_add():
    parent = _make('parent')
    child = parent.add('child', {'attr': 'value'}, extra='extra_value')

    assert len(parent) == 1
    assert parent[0] is child
    assert child.tag == 'child'
    assert child.attrib == {'attr': 'value', 'extra': 'extra_value'}


def test_attr():
    e = _make('test', {'attr1': 'value1'})
    assert e.attr('attr1') == 'value1'
    assert e.attr('nonexistent') == ''
    assert e.attr('nonexistent', 'default') == 'default'


def test_findfirst():
    parent = _make('parent')
    child1 = _make('child1')
    child2 = _make('child2')
    child3 = _make('child3')

    parent.append(child1)
    parent.append(child2)
    parent.append(child3)

    # Test with no arguments - returns first child
    first = parent.findfirst()
    assert first is child1

    # Test with paths
    found = parent.findfirst('child2', 'child3')
    assert found is child2

    # Test when nothing found
    not_found = parent.findfirst('nonexistent')
    assert not_found is None

    # Test empty parent
    empty_parent = _make('empty')
    assert empty_parent.findfirst() is None


def test_textof():
    parent = _make('parent')
    child1 = _make('child1')
    child1.text = 'text1'
    child2 = _make('child2')
    child2.text = 'text2'

    parent.append(child1)
    parent.append(child2)

    text = parent.textof('child1', 'child3')
    assert text == 'text1'

    no_text = parent.textof('nonexistent')
    assert no_text is None


def test_textlist():
    parent = _make('parent')
    child1 = _make('item')
    child1.text = 'text1'
    child2 = _make('item')
    child2.text = 'text2'
    child3 = _make('other')
    child3.text = 'text3'

    parent.append(child1)
    parent.append(child2)
    parent.append(child3)

    # Test without paths - gets all children text
    all_texts = parent.textlist()
    assert 'text1' in all_texts
    assert 'text2' in all_texts
    assert 'text3' in all_texts

    # Test with specific path
    item_texts = parent.textlist('item')
    assert item_texts == ['text1', 'text2']

    # Test deep option
    grandchild = _make('grandchild')
    grandchild.text = 'deep text'
    child1.append(grandchild)

    shallow_texts = parent.textlist('item', deep=False)
    assert 'deep text' not in shallow_texts

    deep_texts = parent.textlist('item', deep=True)
    assert 'deep text' in deep_texts


def test_textdict():
    parent = _make('parent')
    child1 = _make('name')
    child1.text = 'John'
    child2 = _make('age')
    child2.text = '30'

    parent.append(child1)
    parent.append(child2)

    text_dict = parent.textdict()
    assert text_dict == {'name': 'John', 'age': '30'}

    # Test with specific paths
    name_dict = parent.textdict('name')
    assert name_dict == {'name': 'John'}

    # Test deep option
    container = _make('container')
    nested = _make('nested')
    nested.text = 'nested value'
    container.append(nested)
    parent.append(container)

    shallow_dict = parent.textdict('container', deep=False)
    assert 'nested' not in shallow_dict

    deep_dict = parent.textdict('container', deep=True)
    assert 'nested' in deep_dict
    assert deep_dict['nested'] == 'nested value'
