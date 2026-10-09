import gws
import gws.test.util as u


def _root(project_client):
    return u.gws_root(f'''
        client.elements [
            {{ tag "tag_1" }}
            {{ tag "tag_2" options {{ opt_1 "value_1" }} }}
            {{ tag "tag_3" }}
        ]
        projects+ {{
            uid "project_1"
            client {{ {project_client} }}
        }}
    ''')


def _elements(root):
    return root.app.project('project_1').client.elements


def test_inherited_elements():
    root = _root('options { opt_2 "value_2" }')
    els = _elements(root)
    assert [e.tag for e in els] == ['tag_1', 'tag_2', 'tag_3']
    assert els[1].options.get('opt_1') == 'value_1'


def test_inherited_elements_are_new_objects():
    root = _root('options { opt_2 "value_2" }')
    app_els = root.app.client.elements
    for e in _elements(root):
        assert all(e is not a for a in app_els)
        assert e.parent is root.app.project('project_1').client


def test_add_elements():
    root = _root('''
        addElements [
            { tag "tag_4" after "tag_1" }
            { tag "tag_5" before "tag_3" }
            { tag "tag_6" }
        ]
    ''')
    assert [e.tag for e in _elements(root)] == ['tag_1', 'tag_4', 'tag_2', 'tag_5', 'tag_3', 'tag_6']


def test_add_element_replaces_inherited():
    root = _root('''
        addElements [
            { tag "tag_2" options { opt_1 "value_2" } }
        ]
    ''')
    els = _elements(root)
    assert [e.tag for e in els] == ['tag_1', 'tag_3', 'tag_2']
    assert els[2].options.get('opt_1') == 'value_2'


def test_remove_elements():
    root = _root('''
        removeElements [
            { tag "tag_2" }
        ]
    ''')
    assert [e.tag for e in _elements(root)] == ['tag_1', 'tag_3']
