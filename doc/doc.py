"""Doc generator CLI tool"""

import os
import sys
import json

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
            out_dir = f'{options.APP_DIR}/__build/doc/{options.VERSION}'
        if cmd == 'api':
            out_dir = f'{options.APP_DIR}/__build/apidoc/{options.VERSION}'

    opts.outputDir = out_dir
    mkdir(opts.outputDir)

    if cmd == 'build':
        make_config_ref()
        dog.build_html(opts)
        dog.build_pdf(opts)
        return 0

    if cmd == 'server':
        make_config_ref()
        opts.debug = True
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


def make_config_ref():
    def literal(s):
        return f'<i>"{s}"</i>'

    def code(s):
        return f'<code>{s}</code>'

    def typestring(tid):
        typ = specs[tid]
        c = typ['c']
        if c in {'CLASS', 'TYPE', 'ENUM'}:
            queue.append(tid)
            return f"<a href='#{tid}'>{code(tid)}</a>"
        if c == 'DICT':
            return code('dict')
        if c == 'LIST':
            queue.append(typ['tItem'])
            s = typestring(typ['tItem'])
            return f'<b>[</b>{s}<b>]</b>'
        if c == 'ATOM':
            return code(tid)
        if c == 'LITERAL':
            return ' | '.join(literal(s) for s in typ['literalValues'])
        return ''

    def docstring(tid, enum_value=None):
        # @TODO translations
        typ = specs[tid]
        if enum_value:
            return typ['enumDocs'][enum_value]
        return typ.get('doc')

    def head(tid, info):
        return f'<h6 data-url="#{tid}" id="{tid}">{code(info)} {tid}</h6>'

    def table(heads, rows):
        s = '<table><thead><tr>'
        for h in heads:
            s += '<th>' + h + '</th>'
        s += '</tr></thead><tbody>'
        for row in rows:
            s += '<tr>' + ''.join('<td>' + c + '</td>' for c in row) + '</tr>'
        s += '</tbody></table>'
        return s

    def process(tid):
        typ = specs[tid]
        c = typ['c']

        if c == 'CLASS':
            html[tid] = head(tid, 'struct')
            html[tid] += f'<p>{docstring(tid)}</p>'
            html[tid] += table(
                ['property', 'type', ''],
                [
                    [code(pname), typestring(specs[ptid]['tValue']), docstring(ptid)]
                    for pname, ptid in typ['tProperties'].items()
                ]
            )

        if c == 'TYPE':
            target = specs[typ['tTarget']]
            if target['c'] == 'VARIANT':
                html[tid] = head(tid, 'variant type')
                html[tid] += table(
                    ['type', 'target'],
                    [
                        [literal(mname), typestring(mtid)]
                        for mname, mtid in target['tMembers'].items()
                    ]
                )
            else:
                html[tid] = head(tid, "type")
                html[tid] += f'<p>{docstring(tid)}</p>'

        if c == 'ENUM':
            html[tid] = head(tid, 'enumeration')
            html[tid] += table(
                ['value', ''],
                [
                    [literal(key), docstring(tid, key)]
                    for key in typ['enumValues']
                ]
            )

    with open(options.APP_DIR + '/__build/types.json') as fp:
        specs = json.load(fp)['types']

    start = 'gws.base.application.Config'
    queue = [start]
    done = set()
    html = {}

    while queue:
        tid = queue.pop(0)
        if tid in done:
            continue
        done.add(tid)
        process(tid)

    res = html.pop(start)
    res += ''.join(v for _, v in sorted(html.items()))

    with open(options.DOC_DIR + '/books/admin-de/__build.configref.de.html', 'w') as fp:
        fp.write(res)


def mkdir(d):
    dog.util.run(['rm', '-fr', d])
    dog.util.run(['mkdir', '-p', d])


if __name__ == '__main__':
    cli.main('doc', main, USAGE)
