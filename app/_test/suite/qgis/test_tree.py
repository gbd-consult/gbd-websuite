import _test.util as u


def layer_tree_structure(project_info_layer):
    r = {'_uid': project_info_layer['uid']}
    if project_info_layer.get('layers'):
        r['layers'] = [layer_tree_structure(la) for la in project_info_layer['layers']]
    return r


def test_tree_struct_full():
    exp = {
        "_uid": "tree_full.map.t",
        "layers": [
            {
                "_uid": "tree_full.map.groupone",
                "layers": [
                    {
                        "_uid": "tree_full.map.grouponeone",
                        "layers": [
                            {
                                "_uid": "tree_full.map.points_ghana_25832"
                            }
                        ]
                    }
                ]
            },
            {
                "_uid": "tree_full.map.grouptwo",
                "layers": [
                    {
                        "_uid": "tree_full.map.squares_memphis_25832"
                    },
                    {
                        "_uid": "tree_full.map.grouptwoone",
                        "layers": [
                            {
                                "_uid": "tree_full.map.squares_ny_2263"
                            },
                            {
                                "_uid": "tree_full.map.grouptwooneone",
                                "layers": [
                                    {
                                        "_uid": "tree_full.map.squares_dus1_3857"
                                    },
                                    {
                                        "_uid": "tree_full.map.squares_dus2_3857"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }

    r = u.cmd('projectInfo', {'projectUid': 'tree_full'}).json()
    tree = layer_tree_structure(r['project']['map']['layers'][0])
    assert tree == exp


def test_tree_struct_with_filter():
    exp = {
        "_uid": "tree_filtered.map.t",
        "layers": [
            {
                "_uid": "tree_filtered.map.groupone",
                "layers": [
                    {
                        "_uid": "tree_filtered.map.grouponeone",
                        "layers": [
                            {
                                "_uid": "tree_filtered.map.points_ghana_25832"
                            }
                        ]
                    }
                ]
            },
            {
                "_uid": "tree_filtered.map.grouptwooneone",
                "layers": [
                    {
                        "_uid": "tree_filtered.map.squares_dus1_3857"
                    },
                    {
                        "_uid": "tree_filtered.map.squares_dus2_3857"
                    }
                ]
            }
        ]
    }

    r = u.cmd('projectInfo', {'projectUid': 'tree_filtered'}).json()
    tree = layer_tree_structure(r['project']['map']['layers'][0])
    assert tree == exp


def test_tree_struct_with_exclude():

    # NB empty groups are excluded as well

    exp = {
        "_uid": "tree_exclude.map.t",
        "layers": [
            {
                "_uid": "tree_exclude.map.grouptwo",
                "layers": [
                    {
                        "_uid": "tree_exclude.map.squares_memphis_25832"
                    },
                    {
                        "_uid": "tree_exclude.map.grouptwoone",
                        "layers": [
                            {
                                "_uid": "tree_exclude.map.squares_ny_2263"
                            },
                            {
                                "_uid": "tree_exclude.map.grouptwooneone",
                                "layers": [
                                    {
                                        "_uid": "tree_exclude.map.squares_dus2_3857"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }

    r = u.cmd('projectInfo', {'projectUid': 'tree_exclude'}).json()
    tree = layer_tree_structure(r['project']['map']['layers'][0])
    assert tree == exp


def test_tree_struct_with_flatten():
    exp = {
        "_uid": "tree_flattened.map.t",
        "layers": [
            {
                "_uid": "tree_flattened.map.groupone",
                "layers": [
                    {
                        "_uid": "tree_flattened.map.grouponeone"
                    }
                ]
            },
            {
                "_uid": "tree_flattened.map.grouptwo",
                "layers": [
                    {
                        "_uid": "tree_flattened.map.squares_memphis_25832"
                    },
                    {
                        "_uid": "tree_flattened.map.grouptwoone"
                    }
                ]
            }
        ]
    }

    r = u.cmd('projectInfo', {'projectUid': 'tree_flattened'}).json()
    tree = layer_tree_structure(r['project']['map']['layers'][0])
    assert tree == exp
