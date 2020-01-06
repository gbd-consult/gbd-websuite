import _test.util as u

base = '_/cmd/assetHttpGetPath/'
content = '0123456789\n'


def test_std_mime():
    assert u.req(base + 'projectUid/allow_std_mime/path/x.txt').status_code == 200
    assert u.req(base + 'projectUid/allow_std_mime/path/x.pdf').status_code == 200
    assert u.req(base + 'projectUid/allow_std_mime/path/x.png').status_code == 200
    assert u.req(base + 'projectUid/allow_std_mime/path/x.exe').status_code == 404


def test_allow_mime():
    assert u.req(base + 'projectUid/allow_only_pdf/path/x.txt').status_code == 404
    assert u.req(base + 'projectUid/allow_only_pdf/path/x.pdf').status_code == 200
    assert u.req(base + 'projectUid/allow_only_pdf/path/x.png').status_code == 404
    assert u.req(base + 'projectUid/allow_only_pdf/path/x.exe').status_code == 404


def test_deny_mime():
    assert u.req(base + 'projectUid/allow_all_but_pdf/path/x.txt').status_code == 200
    assert u.req(base + 'projectUid/allow_all_but_pdf/path/x.pdf').status_code == 404
    assert u.req(base + 'projectUid/allow_all_but_pdf/path/x.png').status_code == 200
    assert u.req(base + 'projectUid/allow_all_but_pdf/path/x.exe').status_code == 404


def test_content():
    assert u.req(base + 'projectUid/allow_std_mime/path/x.txt').text == f'txt:{content}'
    assert u.req(base + 'projectUid/allow_std_mime/path/x.pdf').text == f'pdf:{content}'
    assert u.req(base + 'projectUid/allow_std_mime/path/x.png').text == f'png:{content}'


def test_asset_from_subdir():
    r = u.req(base + 'projectUid/allow_std_mime?path=subdir/y.xml')
    assert r.status_code == 200
    assert r.text == f'xml:{content}'


def test_dotted_path_not_allowed():
    assert u.req(base + 'projectUid/allow_std_mime?path=./x.txt').status_code == 404
    assert u.req(base + 'projectUid/allow_std_mime?path=subdir/../x.txt').status_code == 404


def test_download_asset():
    r = u.req('_/cmd/assetHttpGetDownload/projectUid/allow_std_mime/path/x.txt')
    assert r.status_code == 200
    assert r.headers['content-disposition'] == 'attachment; filename="x.txt"'
    assert r.text == f'txt:{content}'


def test_download_asset_from_subdir():
    r = u.req('_/cmd/assetHttpGetDownload/projectUid/allow_std_mime?path=subdir/y.xml')
    assert r.status_code == 200
    assert r.headers['content-disposition'] == 'attachment; filename="y.xml"'
    assert r.text == f'xml:{content}'


def test_web_dir():
    assert u.req('/').text == f'index.html:{content}'
    assert u.req('/y.html').text == f'y.html:{content}'
    assert u.req('/subdir/z.html').text == f'z.html:{content}'
