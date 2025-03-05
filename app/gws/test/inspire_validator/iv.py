"""Run INSPIRE validator tests.

Reference:
    - https://inspire.ec.europa.eu/validator/swagger-ui.html
"""

import os
import sys
import re

import jq

LOCAL_APP_DIR = os.path.abspath(os.path.dirname(__file__) + '/../../..')
sys.path.insert(0, LOCAL_APP_DIR)

import gws
import gws.lib.cli as cli
import gws.lib.jsonx as jsonx
import gws.lib.net

USAGE = """
GWS INSPIRE validator
~~~~~~~~~~~~~~~~~~~~~

Commands:

    iv.py dir
        - show available test suites

    iv.py run <suite-filter> <url>
        - run suites matching the filter against the given URL

Options:
    --host   - validator host (default: 127.0.0.1)
    --port   - validator port (default: 8090)
    --raw    - print raw output
    --all    - report all tests
    
The validator docker image must be started before running this script (see `inspire_validator/docker-compose.yml`).
    
"""

OPTIONS = {}


def main(args):
    OPTIONS['host'] = args.get('host', '127.0.0.1')
    OPTIONS['port'] = args.get('port', '8090')
    OPTIONS['raw'] = args.get('raw', False)
    OPTIONS['all'] = args.get('all', False)

    cmd = args.get(1)

    if cmd == 'dir':
        _show_suites()
        return 0

    if cmd == 'run':
        uids = list(_get_suite_uids(args.get(2, '')))
        if not uids:
            cli.fatal('no test suites found')
        url = args.get(3)
        if not url:
            cli.fatal('no --url option')
        ok = _run_tests(uids, url)
        return 0 if ok else 1


##

def _get_object_types():
    res = _invoke('TestObjectTypes.json')
    for q in _get_list('.. | .TestObjectType?', res):
        yield q['id'], q['label'], q['description']


def _get_suite_uids(search):
    for s in _get_suites():
        if s['id'] == search or search.lower() in s['label'].lower():
            yield s['id']


def _show_suites():
    print(cli.text_table(_get_suites(), header='auto'))


def _get_suites():
    types = {t[0]: t[1] for t in _get_object_types()}

    res = _invoke('ExecutableTestSuites.json')

    for q in _get_list('.. | .ExecutableTestSuite?', res):
        href = jq.first('.supportedTestObjectTypes.testObjectType.href', q)
        # like "http://localhost:8090/validator/v2/TestObjectTypes/5a60dded-0cb0-4977-9b06-16c6c2321d2e.json"
        typ = ''
        m = re.search(r'([^/]+)\.json$', href)
        if m:
            typ = types.get('EID' + m.group(1))
        yield dict(id=q['id'], label=q['label'], type=typ)


def _run_tests(suite_uids, url):
    request = {
        "executableTestSuiteIds": suite_uids,
        "label": "test",
        "arguments": {},
        "testObject": {
            "resources": {
                "serviceEndpoint": url
            }
        }
    }
    res = _invoke('TestRuns', method='POST', json=request)
    if res.get('error'):
        cli.fatal(res['error'])

    uid = jq.first('.EtfItemCollection.testRuns.TestRun.id', res)
    if not uid:
        cli.fatal('no test run uid')

    while True:
        gws.u.sleep(3)
        res = _invoke(f'TestRuns/{uid}')
        if res.get('error'):
            cli.fatal(res['error'])
        status = jq.first('.EtfItemCollection.testRuns.TestRun.status', res)
        if status != 'UNDEFINED':
            break
        cli.info(f'test {uid}: waiting...')
        continue

    results = list(_parse_test_results(res))
    cnt_total = len(results)
    failed = [r for r in results if not r['passed']]
    ok = len(failed) == 0

    if not OPTIONS['all']:
        results = failed
    if OPTIONS['raw']:
        print(jsonx.to_pretty_string(results))
        return ok

    cli.info(f'')
    cli.info(f'test: {len(suite_uids)} suite(s), {cnt_total} tests, {len(failed)} failed')
    cli.info(f'')

    for r in results:
        fn = cli.info if r['passed'] else cli.error
        labels = ' -> '.join(s.get('label') or '' for s in r['specs'])
        desc = _unhtml(' '.join(s.get('description') or '' for s in r['specs']))
        if r['message']:
            desc = r['message'] + ' ' + desc
        fn(('passed' if r['passed'] else 'FAILED') + ' ' + labels + ': ' + desc)

    return ok


def _parse_test_results(res):
    all_messages = {}

    for msg in jq.all('.. | .LangTranslationTemplateCollection? | .[]?', res):
        all_messages[msg['name']] = jq.first('.translationTemplates.TranslationTemplate["$"]', msg)
    # the default is just silly
    all_messages['TR.fallbackInfo'] = ''

    all_results = []

    for step in _get_list('.. | .TestStepResult?', res):
        all_results.append(step)
        if 'testAssertionResults' in step:
            all_results.extend(_get_list('.. | .TestAssertionResult?', step))

    for q in all_results:
        messages = []

        for m in _get_list('.messages.message', q):
            msg = all_messages.get(m['ref'], '')
            for a in _get_list('.. | .argument?', m):
                msg = msg.replace('{' + a['token'] + '}', str(a['$']))
            messages.append(msg)

        specs = []
        ref = jq.first('. | .resultedFrom? | .ref', q)

        while ref:
            spec = jq.first(f'.. | select(.id? == "{ref}")', res)
            if not spec:
                break
            specs.insert(0, _pick(spec, 'id', 'label', 'description', 'statementForExecution', 'expression'))
            ref = jq.first('. | .parent? | .ref', spec)

        ref = jq.first('. | .resultedFrom? | .href', q)
        if ref:
            specs.append({'label': ref})

        yield dict(
            id=q['id'],
            passed='PASSED' in q['status'],
            message=', '.join(messages),
            specs=specs
        )


def _invoke(path, **kwargs):
    url = f'http://{OPTIONS["host"]}:{OPTIONS["port"]}/validator/v2/{path}'
    try:
        res = gws.lib.net.http_request(url, headers={'Accept': 'application/json'}, **kwargs)
        res.raise_if_failed()
        return jsonx.from_string(res.text)
    except gws.lib.net.Error as exc:
        try:
            res = jsonx.from_string(exc.args[1])
            cli.error(jsonx.to_pretty_string(res))
        except:
            cli.error(exc)
        cli.fatal(f'HTTP ERROR')


def _get_list(q, where):
    for s in jq.all(q, where):
        if not s:
            continue
        if isinstance(s, list):
            yield from s
            continue
        yield s


def _pick(d, *keys):
    return {k: d.get(k, '') for k in keys}


def _unhtml(s):
    s = re.sub(r'<[^>]+>', '', s)
    return re.sub(r'\s+', ' ', s.strip())


def _pr(r):
    print(jsonx.to_pretty_string(r))


##

if __name__ == '__main__':
    cli.main('test', main, USAGE)
