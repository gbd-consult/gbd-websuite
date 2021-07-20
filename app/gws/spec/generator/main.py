import json
import os
import re

from . import base, manifest, parser, normalizer, strings, typescript

# EXCLUDE_PATHS = ['___', '/vendor/']
EXCLUDE_PATHS = ['/vendor/']

CONST_PATH = base.APP_DIR + '/gws/core/const.py'
STRINGS_PATH = base.APP_DIR + '/gws/spec/strings.ini'
TYPESCRIPT_PATH = base.APP_DIR + '/js/src/gws/core/__build.gwsapi.ts'

BUNDLE_FILENAME = "__build.client.json"
VENDOR_BUNDLE_PATH = base.APP_DIR + '/gws/__build.vendor.js'

SPEC_BUILD_DIR = base.APP_DIR + '/gws/spec/__build'
PLUGIN_DIR = base.APP_DIR + '/gws/plugin'

SYSTEM_CHUNKS = [
    base.Data(name='gws', sourceDir=base.APP_DIR + '/js/src/gws', bundleDir=base.APP_DIR + '/gws'),
    base.Data(name='gws.core', sourceDir=base.APP_DIR + '/gws/core', bundleDir=base.APP_DIR + '/gws'),
    base.Data(name='gws.base', sourceDir=base.APP_DIR + '/gws/base', bundleDir=base.APP_DIR + '/gws'),
    base.Data(name='gws.lib', sourceDir=base.APP_DIR + '/gws/lib', bundleDir=base.APP_DIR + '/gws'),
    base.Data(name='gws.server', sourceDir=base.APP_DIR + '/gws/server', bundleDir=base.APP_DIR + '/gws'),
]


def generate_for_build(manifest_path):
    os.makedirs(SPEC_BUILD_DIR, exist_ok=True)

    meta = _init_meta('build', manifest_path)
    _write_json(SPEC_BUILD_DIR + '/meta.spec.json', meta)

    parser.write_file(
        CONST_PATH,
        re.sub(
            r'VERSION\s*=.*',
            f'VERSION = {meta.version!r}',
            parser.read_file(CONST_PATH)))

    state = base.ParserState()

    parser.parse(state, meta)
    _write_json(SPEC_BUILD_DIR + '/parsed.spec.json', state)

    normalizer.normalize(state, meta)
    _write_json(SPEC_BUILD_DIR + '/full.spec.json', state)

    ts = typescript.generate(state, meta)
    parser.write_file(SPEC_BUILD_DIR + '/api.ts', ts)
    parser.write_file(TYPESCRIPT_PATH, ts)

    specs = normalizer.prepare_for_server(state, meta)
    _write_json(SPEC_BUILD_DIR + '/server.spec.json', specs)

    strs = strings.generate(state, specs, parser.read_file(STRINGS_PATH))
    _write_json(SPEC_BUILD_DIR + '/strings.json', strs)


def generate_for_server(manifest_path):
    meta = _init_meta('server', manifest_path)

    state = base.ParserState()

    parser.parse(state, meta)
    normalizer.normalize(state, meta)
    specs = normalizer.prepare_for_server(state, meta)

    strs = strings.generate(state, specs, parser.read_file(STRINGS_PATH))

    return {
        'meta': _as_json(meta),
        'specs': _as_json(specs),
        'strings': strs,
    }


def _init_meta(mode, manifest_path):
    meta = base.Data(
        mode=mode,
        manifest_path=None,
        manifest=None,
        version=base.VERSION,
    )

    if manifest_path:
        try:
            base.log.debug('loading manifest', manifest_path)
            meta.manifest_path = manifest_path
            meta.manifest = manifest.load(manifest_path)
        except Exception as e:
            raise base.Error(f'error loading manifest {manifest_path!r}') from e

    chunks = list(SYSTEM_CHUNKS) + manifest.enumerate_plugins(meta.manifest, PLUGIN_DIR)

    for chunk in chunks:
        _enum_sources(chunk)

    meta.chunks = chunks

    meta.BUNDLE_FILENAME = BUNDLE_FILENAME
    meta.VENDOR_BUNDLE_PATH = VENDOR_BUNDLE_PATH

    return meta


_KIND_PATTERNS = [
    ['.py', 'python'],
    ['/index.ts', 'ts'],
    ['/index.tsx', 'ts'],
    ['/index.css.js', 'css'],
    ['.theme.css.js', 'theme'],
    ['/strings.ini', 'strings'],
]


def _enum_sources(chunk):
    excl = EXCLUDE_PATHS + (chunk.exclude or [])
    chunk.paths = {key: [] for _, key in _KIND_PATTERNS}

    if not os.path.isdir(chunk.sourceDir):
        return

    for path in _find_files(chunk.sourceDir):
        if any(x in path for x in excl):
            continue
        for pattern, key in _KIND_PATTERNS:
            if path.endswith(pattern):
                chunk.paths[key].append(path)
                break


def _find_files(where):
    for fname in os.listdir(where):
        if fname.startswith('.'):
            continue
        path = os.path.join(where, fname)
        if os.path.isdir(path):
            yield from _find_files(path)
            continue
        yield path


def _as_json(x):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, bytes):
        return repr(x)
    if isinstance(x, (list, tuple)):
        return [_as_json(x) for x in x]
    if isinstance(x, dict):
        return {k: _as_json(v) for k, v in x.items()}
    d = _as_json(vars(x))
    d['_'] = type(x).__name__
    return d


def _write_json(path, obj):
    parser.write_file(path, json.dumps(_as_json(obj), indent=4, sort_keys=True))
