"""Tests for the source module"""

import gws
import gws.test.util as u
import gws.gis.source as source

import gws.lib.crs as crs


def test_layer_matches():
    lf = source.LayerFilter(level=1,
                            names=['n1', 'n2', 'n3'],
                            titles=['t1', 't2', 't3'],
                            pattern='test',
                            isGroup=True,
                            isImage=True,
                            isQueryable=True,
                            isVisible=True)

    sl = gws.SourceLayer(aLevel=1,
                         name='n1',
                         title='t1',
                         aPath='/home/test/sl1',
                         isGroup=True,
                         isImage=True,
                         isQueryable=True,
                         isVisible=True)
    assert source.layer_matches(sl, lf)
    sl.aLevel = 2
    assert not source.layer_matches(sl, lf)
    sl.aLevel = 1
    sl.name = 'n2'
    assert source.layer_matches(sl, lf)
    sl.name = 'n1'
    sl.title = 't4'
    assert not source.layer_matches(sl, lf)
    sl.title = 't1'
    sl.apath = '/foo/bar'
    lf.pattern = 'something'
    assert not source.layer_matches(sl, lf)
    lf.pattern = 'test'
    sl.aPath = '/home/test/sl1'
    sl.isGroup = False
    assert not source.layer_matches(sl, lf)
    sl.isGroup = True
    sl.isImage = False
    assert not source.layer_matches(sl, lf)
    sl.isImage = True
    sl.isVisible = False
    assert not source.layer_matches(sl, lf)
    sl.isVisible = True
    sl.isQueryable = False
    assert not source.layer_matches(sl, lf)
    sl.isQueryable = True
    lf = source.LayerFilter()
    assert source.layer_matches(sl, lf)


def test_check_layers():
    sl1 = gws.SourceLayer(name='n1')
    sl2 = gws.SourceLayer(metadata=gws.Metadata(title='t2'))
    sl3 = gws.SourceLayer(name='n3')
    sl4 = gws.SourceLayer(metadata=gws.Metadata(title='t4'))
    sl5 = gws.SourceLayer(name='n5')
    sl6 = gws.SourceLayer(metadata=gws.Metadata(title='t6'))

    layers1 = [sl1, sl2]
    layers2 = [sl3, sl4]
    layers3 = [sl5, sl6]

    sl1.layers = layers2
    sl3.layers = layers3

    test = source.check_layers(layers1)

    assert test[0].get('name') == 'n1'
    assert test[0].get('layers')[0].get('name') == 'n3'
    assert test[0].get('layers')[0].get('layers')[0].get('name') == 'n5'
    assert test[0].get('layers')[0].get('layers')[0].get('aUid') == 'n5'
    assert test[0].get('layers')[0].get('layers')[0].get('aPath') == '/n1/n3/n5'
    assert test[0].get('layers')[0].get('layers')[0].get('aLevel') == 3
    assert not test[0].get('layers')[0].get('layers')[0].get('layers')

    assert test[0].get('layers')[0].get('layers')[1].get('metadata').get('title') == 't6'
    assert test[0].get('layers')[0].get('layers')[1].get('aUid') == 't6'
    assert test[0].get('layers')[0].get('layers')[1].get('aPath') == '/n1/n3/t6'
    assert test[0].get('layers')[0].get('layers')[1].get('aLevel') == 3
    assert test[0].get('layers')[0].get('layers')[1].get('layers') == []

    assert test[0].get('layers')[0].get('aUid') == 'n3'
    assert test[0].get('layers')[0].get('aPath') == '/n1/n3'
    assert test[0].get('layers')[0].get('aLevel') == 2

    assert test[0].get('layers')[1].get('metadata').get('title') == 't4'
    assert test[0].get('layers')[1].get('aPath') == '/n1/t4'
    assert test[0].get('layers')[1].get('aLevel') == 2
    assert test[0].get('layers')[1].get('layers') == []

    assert test[0].get('aUid') == 'n1'
    assert test[0].get('aPath') == '/n1'
    assert test[0].get('aLevel') == 1

    assert test[1].get('metadata').get('title') == 't2'
    assert test[1].get('aPath') == '/t2'
    assert test[1].get('aLevel') == 1
    assert test[1].get('aUid') == 't2'
    assert test[1].get('layers') == []


def test_check_layers_revert():
    sl1 = gws.SourceLayer(name='n1')
    sl2 = gws.SourceLayer(metadata=gws.Metadata(title='t2'))
    sl3 = gws.SourceLayer(name='n3')
    sl4 = gws.SourceLayer(metadata=gws.Metadata(title='t4'))
    sl5 = gws.SourceLayer(name='n5')
    sl6 = gws.SourceLayer(metadata=gws.Metadata(title='t6'))

    layers1 = [sl1, sl2]
    layers2 = [sl3, sl4]
    layers3 = [sl5, sl6]

    sl1.layers = layers2
    sl3.layers = layers3

    test = source.check_layers(layers1, revert=True)

    assert test[0].get('metadata').get('title') == 't2'
    assert test[0].get('aUid') == 't2'
    assert test[0].get('aPath') == '/t2'
    assert test[0].get('aLevel') == 1
    assert test[0].get('layers') == []

    assert test[1].get('layers')[0].get('metadata').get('title') == 't4'
    assert test[1].get('layers')[0].get('aUid') == 't4'
    assert test[1].get('layers')[0].get('aPath') == '/n1/t4'
    assert test[1].get('layers')[0].get('aLevel') == 2
    assert test[1].get('layers')[0].get('layers') == []

    assert test[1].get('layers')[1].get('name') == 'n3'
    assert test[1].get('layers')[1].get('layers')[0].get('metadata').get('title') == 't6'
    assert test[1].get('layers')[1].get('layers')[0].get('aUid') == 't6'
    assert test[1].get('layers')[1].get('layers')[0].get('aPath') == '/n1/n3/t6'
    assert test[1].get('layers')[1].get('layers')[0].get('aLevel') == 3
    assert test[1].get('layers')[1].get('layers')[0].get('layers') == []

    assert test[1].get('layers')[1].get('layers')[1].get('name') == 'n5'
    assert test[1].get('layers')[1].get('layers')[1].get('aUid') == 'n5'
    assert test[1].get('layers')[1].get('layers')[1].get('aPath') == '/n1/n3/n5'
    assert test[1].get('layers')[1].get('layers')[1].get('aLevel') == 3
    assert test[1].get('layers')[1].get('layers')[1].get('layers') == []

    assert test[1].get('layers')[1].get('aPath') == '/n1/n3'
    assert test[1].get('layers')[1].get('aLevel') == 2
    assert test[1].get('aUid') == 'n1'
    assert test[1].get('aPath') == '/n1'
    assert test[1].get('aLevel') == 1


def test_filter_layers():
    lf = source.LayerFilter(names=['n2', 'n4'],
                            isGroup=True,
                            isImage=True,
                            isQueryable=True,
                            isVisible=True)

    sl1 = gws.SourceLayer(aLevel=1,
                          name='n1',
                          isGroup=True,
                          isImage=True,
                          isQueryable=True,
                          isVisible=True)

    sl2 = gws.SourceLayer(aLevel=1,
                          name='n2',
                          isGroup=True,
                          isImage=True,
                          isQueryable=True,
                          isVisible=True)

    sl3 = gws.SourceLayer(aLevel=2,
                          name='n3',
                          isGroup=True,
                          isImage=True,
                          isQueryable=True,
                          isVisible=True)

    sl4 = gws.SourceLayer(aLevel=2,
                          name='n4',
                          isGroup=True,
                          isImage=True,
                          isQueryable=True,
                          isVisible=True)

    layers1 = [sl1, sl2]
    layers2 = [sl3, sl4]

    sl1.layers = layers2

    layers1 = source.check_layers(layers1)

    test = source.filter_layers(layers1, lf)

    assert test[0].get('aLevel') == 1
    assert test[0].get('name') == 'n2'
    assert test[0].get('isGroup')
    assert test[0].get('isImage')
    assert test[0].get('isQueryable')
    assert test[0].get('isVisible')
    assert test[0].get('aUid') == 'n2'
    assert test[0].get('aPath') == '/n2'
    assert not test[0].get('layers')

    assert test[1].get('aLevel') == 2
    assert test[1].get('name') == 'n4'
    assert test[1].get('isGroup')
    assert test[1].get('isImage')
    assert test[1].get('isQueryable')
    assert test[1].get('isVisible')
    assert test[1].get('aUid') == 'n4'
    assert test[1].get('aPath') == '/n1/n4'
    assert not test[1].get('layers')


def test_combined_crs_list():
    sl1 = gws.SourceLayer(supportedCrs=[])
    sl2 = gws.SourceLayer(supportedCrs=[crs.WGS84])
    sl3 = gws.SourceLayer(supportedCrs=[crs.WGS84, crs.WEBMERCATOR])
    sl4 = gws.SourceLayer(supportedCrs=[crs.WGS84])
    sl5 = gws.SourceLayer(supportedCrs=[])
    sl6 = gws.SourceLayer(supportedCrs=[crs.WGS84, crs.WEBMERCATOR])
    assert source.combined_crs_list([sl1, sl2, sl3, sl4, sl5, sl6]) == [crs.WGS84]
