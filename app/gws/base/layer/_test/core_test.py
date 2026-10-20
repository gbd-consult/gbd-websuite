import gws
import gws.lib.crs
import gws.base.layer.core as core

CRS_3857 = gws.lib.crs.get(3857)
CRS_25832 = gws.lib.crs.get(25832)


def _layer():
    la = core.Object()
    la.mapCrs = CRS_3857
    la.grabbers = {3857: 'grabber_1', 25832: 'grabber_2'}
    return la


def test_grabber_for_map_crs():
    assert _layer().grabber_for(gws.LayerRenderInput()) == 'grabber_1'


def test_grabber_for_target_crs():
    assert _layer().grabber_for(gws.LayerRenderInput(targetCrs=CRS_25832)) == 'grabber_2'


def test_grabber_for_unknown_crs():
    assert _layer().grabber_for(gws.LayerRenderInput(targetCrs=gws.lib.crs.get(4326))) is None
