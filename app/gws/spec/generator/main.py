import os

from .. import core
from . import base, configref, manifest, normalizer, parser, extractor, strings, typescript, util

Error = base.Error


def generate_and_write(root_dir='', out_dir='', manifest_path='', debug=False):
    base.log.set_level('DEBUG' if debug else 'INFO')

    gen = _run_generator(root_dir, out_dir, manifest_path, debug)

    to_path(gen.specData, out_dir + '/specs.json')

    util.write_file(gen.outDir + '/gws.generated.ts', gen.typescript)

    util.write_file(gen.outDir + '/configref.en.md', gen.configRef['en'])
    util.write_file(gen.outDir + '/configref.de.md', gen.configRef['de'])


def generate(manifest_path='') -> core.SpecData:
    gen = _run_generator(manifest_path=manifest_path)
    return gen.specData


def to_path(specs: core.SpecData, path: str):
    util.write_json(
        path,
        {
            'meta': specs.meta,
            'chunks': specs.chunks,
            'serverTypes': specs.serverTypes,
            'strings': specs.strings,
        },
    )
    return path


def from_path(path: str) -> core.SpecData:
    d = util.read_json(path)
    s = core.SpecData()
    s.meta = d['meta']
    s.chunks = [core.Chunk(**c) for c in d['chunks']]
    s.serverTypes = [core.make_type(t) for t in d['serverTypes']]
    s.strings = d['strings']
    return s


def _run_generator(root_dir='', out_dir='', manifest_path='', debug=False):
    gen = base.Generator()
    gen.rootDir = root_dir or base.v.APP_DIR
    gen.outDir = out_dir
    gen.selfDir = base.v.SELF_DIR
    gen.debug = debug
    gen.manifestPath = manifest_path

    _init_generator(gen)
    gen.dump('000_init')

    parser.parse(gen)
    gen.dump('001_parsed')

    normalizer.normalize(gen)
    gen.dump('002_normalized')

    extractor.extract(gen)
    gen.dump('003_extracted')

    gen.typescript = typescript.create(gen)
    gen.strings = strings.collect(gen)

    gen.configRef['en'] = configref.create(gen, 'en')
    gen.configRef['de'] = configref.create(gen, 'de')

    gen.specData = core.SpecData()
    gen.specData.meta = gen.meta
    gen.specData.chunks = gen.chunks
    gen.specData.serverTypes = gen.serverTypes
    gen.specData.strings = gen.strings

    return gen


def _init_generator(gen: base.Generator):
    gen.meta = {
        'version': util.read_file(gen.rootDir + '/VERSION').strip(),
        'manifestPath': None,
        'manifest': None,
    }

    def _chunk(name, source_dir, bundle_dir):
        cc = core.Chunk()
        cc.name = name
        cc.sourceDir = source_dir
        cc.bundleDir = bundle_dir
        cc.paths = {kind: [] for _, kind in base.v.FILE_KINDS}
        cc.exclude = []
        return cc

    gen.chunks = []

    for name, path in base.v.SYSTEM_CHUNKS:
        cc = _chunk(
            name,
            gen.rootDir + path,
            base.v.APP_DIR,
        )
        gen.chunks.append(cc)

    manifest_plugins = None

    if gen.manifestPath:
        try:
            base.log.debug(f'loading manifest {gen.manifestPath!r}')
            gen.meta['manifestPath'] = gen.manifestPath
            gen.meta['manifest'] = manifest.from_path(gen.manifestPath)
        except Exception as exc:
            raise base.GeneratorError(f'error loading manifest {gen.manifestPath!r}') from exc
        manifest_plugins = gen.meta['manifest'].get('plugins')

    plugin_dict = {}

    # our plugins
    for path in util.find_dirs(gen.rootDir + base.v.PLUGIN_DIR):
        name = os.path.basename(path)
        cc = _chunk(
            base.v.PLUGIN_PREFIX + '.' + name,
            path,
            path,
        )
        plugin_dict[cc.name] = cc

    # manifest plugins
    for p in manifest_plugins or []:
        path = p.get('path')
        name = p.get('name') or os.path.basename(path)
        if not os.path.isdir(path):
            raise base.GeneratorError(f'error loading plugin {name!r}: directory {path!r} not found')
        cc = _chunk(
            base.v.PLUGIN_PREFIX + '.' + name,
            path,
            path,
        )
        plugin_dict[cc.name] = cc

    gen.chunks.extend(plugin_dict.values())

    for chunk in gen.chunks:
        if not os.path.isdir(chunk.sourceDir):
            continue

        excl = base.v.EXCLUDE_PATHS + (chunk.exclude or [])
        
        for path in util.find_files(chunk.sourceDir):
            if any(x in path for x in excl):
                continue
            for pattern, kind in base.v.FILE_KINDS:
                if path.endswith(pattern):
                    chunk.paths[kind].append(path)
                    break

    root_cc = _chunk(
        'gws',
        gen.rootDir + '/gws',
        gen.rootDir + '/gws',
    )
    root_cc.paths['python'] = [gen.rootDir + '/gws/__init__.py']
    gen.chunks.insert(0, root_cc)
