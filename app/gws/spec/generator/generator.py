import os

from . import base, manifest, normalizer, parser, specs, strings, typescript, util

Error = base.Error


def generate_and_store(root_dir=None, out_dir=None, manifest_path=None, debug=False):
    gen = generate_all(root_dir, out_dir, manifest_path, debug)

    util.write_json(gen.outDir + '/types.json', {
        'meta': gen.meta,
        'types': gen.types
    })

    util.write_json(gen.outDir + '/specs.json', {
        'meta': gen.meta,
        'chunks': gen.chunks,
        'specs': gen.specs,
        'strings': gen.strings
    })

    util.write_file(gen.outDir + '/specs.strings.ini', util.make_ini(gen.strings))
    util.write_file(gen.outDir + '/specs.ts', gen.typescript)


def generate_specs(manifest_path=None):
    gen = generate_all(manifest_path=manifest_path)
    return {
        'meta': gen.meta,
        'chunks': gen.chunks,
        'specs': gen.specs,
        'strings': gen.strings
    }


def generate_all(root_dir=None, out_dir=None, manifest_path=None, debug=False):
    base.log.set_level('DEBUG' if debug else 'INFO')

    gen = base.Generator()
    gen.rootDir = root_dir or base.APP_DIR
    gen.outDir = out_dir
    gen.selfDir = base.SELF_DIR
    gen.debug = debug
    gen.manifestPath = manifest_path

    init_generator(gen)

    gen.dump('init')

    parser.parse(gen)
    gen.dump('parsed')

    normalizer.normalize(gen)
    gen.dump('normalized')

    gen.specs = specs.extract(gen)
    gen.dump('specs')

    gen.typescript = typescript.create(gen)
    gen.strings = strings.collect(gen)

    return gen


def init_generator(gen) -> base.Generator:
    gen.meta = {
        'version': util.read_file(gen.rootDir + '/VERSION').strip(),
        'manifestPath': None,
        'manifest': None,
    }

    gen.chunks = [
        dict(name=name, sourceDir=gen.rootDir + path, bundleDir=base.APP_DIR)
        for name, path in base.SYSTEM_CHUNKS
    ]

    manifest_plugins = None

    if gen.manifestPath:
        try:
            base.log.debug(f'loading manifest {gen.manifestPath!r}')
            gen.meta['manifestPath'] = gen.manifestPath
            gen.meta['manifest'] = manifest.from_path(gen.manifestPath)
        except Exception as e:
            raise base.Error(f'error loading manifest {gen.manifestPath!r}') from e
        manifest_plugins = gen.meta['manifest'].get('plugins')

    plugin_dict = {}

    for path in util.find_dirs(gen.rootDir + base.PLUGIN_DIR):
        name = os.path.basename(path)
        chunk = dict(name=base.PLUGIN_PREFIX + '.' + name, sourceDir=path, bundleDir=path)
        plugin_dict[chunk['name']] = chunk

    for p in manifest_plugins or []:
        path = p.get('path')
        name = p.get('name') or os.path.basename(path)
        chunk = dict(name=base.PLUGIN_PREFIX + '.' + name, sourceDir=path, bundleDir=path)
        plugin_dict[chunk['name']] = chunk

    gen.chunks.extend(plugin_dict.values())

    for chunk in gen.chunks:
        excl = base.EXCLUDE_PATHS + chunk.get('exclude', [])
        chunk['paths'] = {kind: [] for _, kind in base.FILE_KINDS}

        if not os.path.isdir(chunk['sourceDir']):
            continue

        for path in util.find_files(chunk['sourceDir']):
            if any(x in path for x in excl):
                continue
            for pattern, kind in base.FILE_KINDS:
                if path.endswith(pattern):
                    chunk['paths'][kind].append(path)
                    break

    return gen
