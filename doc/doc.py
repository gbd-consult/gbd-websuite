"""Doc generator CLI tool"""

import os
import sys
import json
import re

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../app'))
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../app/gws/lib/vendor'))

import gws.lib.cli as cli

import options
import gws.lib.vendor.dog as dog

import pydoctor.options
import pydoctor.driver

USAGE = """
GWS Doc Builder
~~~~~~~~~~~~~~~
  
    python3 doc.py <command> <options>

Commands:

    build  - generate docs
    server - start the dev server  
    api    - generate api docs

Options:

    -out <dir>
        output directory

    -manifest <path>
        path to MANIFEST.json
        
    -v
        verbose logging
"""


def main(args):
    opts = dog.util.to_data(options.OPTIONS)
    opts.verbose = args.get('v')

    cmd = args.get(1)

    out_dir = args.get('out')
    if not out_dir:
        if cmd in {'build', 'server'}:
            out_dir = f'{opts.buildDir}/doc/{options.VERSION}'
        if cmd == 'api':
            out_dir = f'{opts.buildDir}/apidoc/{options.VERSION}'

    opts.outputDir = out_dir
    mkdir(opts.outputDir)

    if cmd == 'build':
        make_config_ref(opts, 'de')
        dog.build_html(opts)
        dog.build_pdf(opts)
        return 0

    if cmd == 'server':
        make_config_ref(opts, 'de')
        opts.verbose = True
        dog.start_server(opts)
        return 0

    if cmd == 'api':
        make_api(opts)
        return 0

    cli.fatal('invalid arguments, try doc.py -h for help')


def make_api(opts):
    copy_dir = '/tmp/app-copy'
    mkdir(copy_dir)

    rsync = ['rsync', '--archive', '--no-links']
    for e in opts.pydoctorExclude:
        rsync.append('--exclude')
        rsync.append(e)

    rsync.append(options.APP_DIR + '/gws')
    rsync.append(copy_dir)

    dog.util.run(rsync)

    args = list(opts.pydoctorArgs)
    args.extend([
        '--html-output', opts.outputDir,
        '--project-base-dir',
        copy_dir + '/gws',
        copy_dir + '/gws',
    ])

    ps = pydoctor.driver.get_system(pydoctor.options.Options.from_args(args))
    pydoctor.driver.make(ps)

    dog.util.run(['cp', opts.pydoctorExtraCss, opts.outputDir + '/extra.css'])
    dog.util.run(['rm', '-fr', copy_dir])


def make_config_ref(opts, lang):
    STRINGS = {
        'head_property': 'property',
        'head_type': 'type',
        'head_default': 'default',
        'head_value': 'value',

        'head_required': 'required',

        'tag_variant': 'variant',
        'tag_struct': 'struct',
        'tag_enum': 'enumeration',
        'tag_type': 'type',
    }

    nl = '\n'.join

    def escape(s):
        return dog.markdown.escape(str(s))

    def as_literal(s):
        return f'<code class="configref_literal">{escape(s)}</code>'

    def as_typename(s):
        return f'<code class="configref_typename">{escape(s)}</code>'

    def as_category(s):
        return f'<code class="configref_category">{escape(s)}</code>'

    def as_propname(s):
        return f'<code class="configref_propname">{escape(s)}</code>'

    def as_required(s):
        return f'<code class="configref_required">{escape(s)}</code>'

    def as_code(s):
        return f'<code>{escape(s)}</code>'

    def typestring(tid):
        typ = specs[tid]
        c = typ['c']
        if c in {'CLASS', 'TYPE', 'ENUM'}:
            queue.append(tid)
            return f"<a href='#{tid}'>{as_typename(tid)}</a>"
        if c == 'DICT':
            return as_code('dict')
        if c == 'LIST':
            s = typestring(typ['tItem'])
            return '<b>[</b>' + s + '<b>]</b>'
        if c == 'ATOM':
            return as_typename(tid)
        if c == 'LITERAL':
            return ' | '.join(as_literal(s) for s in typ['literalValues'])
        return ''

    def default(tid):
        typ = specs[tid]
        val = typ['tValue']

        if val in specs and specs[val]['c'] == 'LITERAL':
            return ''

        if not typ['hasDefault']:
            return ''

        v = typ['default']
        if v is None or v == '':
            return ''
        return as_literal(v)

    def raw_docstring(tid, enum_value=None):
        # @TODO translations
        typ = specs[tid]
        if enum_value:
            return typ['enumDocs'][enum_value]
        return typ.get('doc')

    def docstring(tid, enum_value=None):
        ds = raw_docstring(tid, enum_value)
        ds = escape(ds)
        ds = re.sub(r'`+(.+?)`+', r'<code>\1</code>', ds)
        return ds

    def head(tid, category):
        return f'<h5 data-url="#{tid}" id="{tid}">{as_category(category)} {tid}</h5>'

    def table(heads, rows):
        head = '<tr>' + nl(f'<th>{h}</th>' for h in heads) + '</tr>'
        body = nl(
            '<tr>' + nl(f'<td>{c}</td>' for c in row) + '</tr>'
            for row in rows
        )
        return f'<table><thead>{head}</thead><tbody>{body}</tbody></table>'

    def process(tid):
        typ = specs[tid]
        c = typ['c']

        if c == 'CLASS':
            html[tid] = head(tid, STRINGS['tag_struct'])
            html[tid] += f'<p>{docstring(tid)}</p>'

            rows = {False: [], True: []}
            for prop_name, prop_tid in typ['tProperties'].items():
                prop_typ = specs[prop_tid]
                rows[prop_typ['hasDefault']].append([
                    as_propname(prop_name) + (as_required('*') if not prop_typ['hasDefault'] else ''),
                    typestring(prop_typ['tValue']),
                    default(prop_tid),
                    docstring(prop_tid),
                ])

            html[tid] += table(
                [STRINGS['head_property'], STRINGS['head_type'], STRINGS['head_default'], ''],
                rows[False] + rows[True],
            )

        if c == 'TYPE':
            target = specs[typ['tTarget']]
            if target['c'] == 'VARIANT':
                html[tid] = head(tid, STRINGS['tag_variant'])
                html[tid] += table(
                    [STRINGS['head_type'], ''],
                    [
                        [
                            as_literal(member_name),
                            typestring(member_tid)
                        ]
                        for member_name, member_tid in target['tMembers'].items()
                    ]
                )
            else:
                html[tid] = head(tid, STRINGS['tag_type'])
                html[tid] += f'<p>{docstring(tid)}</p>'

        if c == 'ENUM':
            html[tid] = head(tid, STRINGS['tag_enum'])
            html[tid] += table(
                [STRINGS['head_value'], ''],
                [
                    [
                        as_literal(key),
                        docstring(tid, key)
                    ]
                    for key in typ['enumValues']
                ]
            )

    def process_all():
        while queue:
            tid = queue.pop(0)
            if tid in done:
                continue
            done.add(tid)
            process(tid)

    specs = json.loads(dog.util.read_file(f'{opts.buildDir}/types.json'))['types']
    start = 'gws.base.application.Config'
    queue = [start]
    done = set()
    html = {}

    process_all()

    res = html.pop(start)
    res += ''.join(v for _, v in sorted(html.items()))

    dog.util.write_file(
        f'{opts.docDir}/books/admin-{lang}/__build.configref.{lang}.html',
        res
    )


def mkdir(d):
    dog.util.run(['rm', '-fr', d])
    dog.util.run(['mkdir', '-p', d])


if __name__ == '__main__':
    cli.main('doc', main, USAGE)
