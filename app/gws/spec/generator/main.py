import json
import os
import re

from . import base, parser, normalizer, typescript

APP_DIR = os.path.abspath(os.path.dirname(__file__) + '/../../..')

# EXCLUDE_PATHS = ['___', '/vendor/']
EXCLUDE_PATHS = ['/vendor/']

CONST_PATH = APP_DIR + '/gws/core/const.py'
STRINGS_PATH = APP_DIR + '/gws/spec/strings.ini'
TYPESCRIPT_PATH = APP_DIR + '/js/src/gws/core/__build.gwsapi.ts'

BUNDLE_FILENAME = "__build.client.json"
VENDOR_BUNDLE_PATH = APP_DIR + '/gws/___build.vendor.js'

SPEC_BUILD_DIR = APP_DIR + '/gws/spec/___build'
PLUGINS_DIR = APP_DIR + '/gws/plugins'


CHUNKS = [
    base.Data(name='gws.core', sourceDir=APP_DIR + '/gws/core', bundleDir=APP_DIR + '/gws'),
    base.Data(name='gws.base', sourceDir=APP_DIR + '/gws/base', bundleDir=APP_DIR + '/gws'),
    base.Data(name='gws.lib', sourceDir=APP_DIR + '/gws/lib', bundleDir=APP_DIR + '/gws'),
    base.Data(name='gws.server', sourceDir=APP_DIR + '/gws/server', bundleDir=APP_DIR + '/gws'),
    base.Data(name='gws', sourceDir=APP_DIR + '/js/src/gws', bundleDir=APP_DIR + '/gws'),
]


def generate_for_development(options):
    options = _prepare(options)
    options.mode = 'dev'

    os.makedirs(SPEC_BUILD_DIR, exist_ok=True)

    if not options.version:
        try:
            options.version = parser.read_file(APP_DIR + '/../VERSION')
        except:
            options.version = ''

    _write_json(SPEC_BUILD_DIR + '/options.spec.json', options)

    parser.write_file(
        CONST_PATH,
        re.sub(
            r'VERSION\s*=.*',
            f'VERSION = {options.version!r}',
            parser.read_file(CONST_PATH)))

    specs = parser.parse(options)
    specs = normalizer.normalize(specs, options)

    _write_json(SPEC_BUILD_DIR + '/full.spec.json', specs)

    ts = typescript.generate(specs, options)
    parser.write_file(SPEC_BUILD_DIR + '/api.ts', ts)
    parser.write_file(TYPESCRIPT_PATH, ts)

    strings = parser.read_file(STRINGS_PATH)

    specs = normalizer.prepare_for_server(specs, options)

    _write_json(SPEC_BUILD_DIR + '/server.spec.json', specs)


def generate_for_server(options):
    options = _prepare(options)
    options.mode = 'server'

    specs = parser.parse(options)
    specs = normalizer.normalize(specs, options)
    specs = normalizer.prepare_for_server(specs, options)

    return {
        'options': options,
        'specs': specs,
    }


def _prepare(options):
    if isinstance(options, dict):
        options = base.Data(**options)
    elif not options:
        options = base.Data()

    if options.manifest:
        options.manifest_path = options.manifest
        try:
            options.manifest = _load_manifest(options.manifest_path)
        except Exception as e:
            raise base.Error(f'error loading manifest {options.manifest_path!r}') from e
    else:
        options.manifest_path = ''
        options.manifest = None

    chunks = list(CHUNKS)

    if options.manifest:
        for p in options.manifest.get('plugins', []):
            path = p.get('path')
            name = p.get('name') or os.path.basename(path)
            chunks.append(base.Data(name=f'gws.plugins.{name}', sourceDir=path, bundleDir=path))
    else:
        for path in _find_dirs(PLUGINS_DIR):
            name = os.path.basename(path)
            chunks.append(base.Data(name=f'gws.plugins.{name}', sourceDir=path, bundleDir=path))

    for chunk in chunks:
        _enum_sources(chunk)

    options.chunks = chunks

    options.BUNDLE_FILENAME = BUNDLE_FILENAME
    options.VENDOR_BUNDLE_PATH = VENDOR_BUNDLE_PATH

    return options


def _load_manifest(path):
    """Load the manifest from the path"""

    # similar to js/helpers/build.js/loadManifest

    def cvt(val, key=''):
        if isinstance(val, str):
            val = _replace_env(val)
            if key.lower().endswith('path') and val.startswith('.'):
                val = os.path.abspath(os.path.join(os.path.dirname(path), val))
            return val
        if isinstance(val, list):
            return [cvt(e) for e in val]
        if isinstance(val, dict):
            return {k: cvt(v, k) for k, v in val.items()}
        return val

    text = re.sub(r'//.*', '', parser.read_file(path))
    return cvt(json.loads(text))


def _replace_env(s):
    def _env(m):
        key = m[1]
        if key in os.environ:
            return os.environ[key]
        raise ValueError(f'unknown variable {key!r} in {s!r}')

    return re.sub(r'\${(\w+)}', _env, s)


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


def _find_dirs(where):
    for fname in os.listdir(where):
        if fname.startswith('.'):
            continue
        path = os.path.join(where, fname)
        if os.path.isdir(path):
            yield path


def _write_json(path, obj):
    def default(o):
        try:
            return vars(o)
        except TypeError:
            return 'json:' + repr(type(o))

    parser.write_file(path, json.dumps(obj, default=default, indent=4, sort_keys=True))
