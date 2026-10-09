import gws
import gws.config.loader as loader
import gws.test.util as u


def test_real_manifest_path_existing(tmp_path):
    p = tmp_path / 'MANIFEST.json'
    p.write_text('{}')
    assert loader.real_manifest_path(str(p)) == str(p)


def test_real_manifest_path_missing(tmp_path):
    with u.raises(gws.Error):
        loader.real_manifest_path(str(tmp_path / 'MANIFEST.json'))
