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

    main.py dir
        - show available test suites

    main.py run <suite-filter> <url>
        - run suites matching the filter against the given URL

Options:
    --host                - validator host (default: 127.0.0.1)
    --port                - validator port (default: 8090)
    --all                 - report all tests (default: only failed)
    --save-parsed <path>  - save parsed output
    --save-raw <path>     - save raw output
    
The validator docker container must be started before running this script (see `inspire_validator/docker-compose.yml`).
    
"""

OPTIONS = {}


def main(args):
    OPTIONS['host'] = args.get('host', '127.0.0.1')
    OPTIONS['port'] = args.get('port', '8090')
    OPTIONS['save-raw'] = args.get('save-raw', '')
    OPTIONS['save-parsed'] = args.get('save-parsed', '')
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
    cli.info(f'running suites {suite_uids}')
    res = _invoke_tests(suite_uids, url)
    return _save_and_report(res)


def _invoke_tests(suite_uids, url):
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
            return res
        cli.info(f'test {uid}: waiting...')
        continue


def _save_and_report(res):
    if OPTIONS['save-raw']:
        jsonx.to_path(OPTIONS['save-raw'], res, pretty=True)

    results = _parse_test_results(res)
    stats = f'TOTAL={len(results)}'

    by_status = {}
    for r in results:
        by_status[r['status']] = by_status.get(r['status'], 0) + 1
    for k, v in sorted(by_status.items()):
        stats += f' {k}={v}'

    if not OPTIONS['all']:
        results = [r for r in results if r['status'] != 'passed']

    if OPTIONS['save-parsed']:
        jsonx.to_path(OPTIONS['save-parsed'], results, pretty=True)

    cli.info(f'')
    cli.info(stats)
    cli.info(f'')

    for r in results:
        _report_result(r)

    return by_status.get('passed', 0) == len(results)


def _report_result(r):
    fn = cli.error if r['status'] == 'FAILED' else cli.info

    fn(r['id'])

    msg = r['status']
    if r['message']:
        msg += ': ' + r['message']
    fn(msg)

    if r['specs']:
        _report_expressions(r['specs'][0], fn)

        for s in r['specs']:
            label = s.get('label', 'UNKNOWN')
            # desc = _unhtml(s.get('description', ''))
            # if desc:
            #     label += f' ({desc})'
            fn('>> ' + label)

    fn('')
    fn('=' * 80)
    fn('')


def _report_expressions(s, fn):
    q = s.get('expression', '')
    if q:
        fn('| ')
        for ln in str(q).splitlines():
            fn('| ' + ln)

    q = s.get('statementForExecution', '')
    if q and q != 'NOT_APPLICABLE':
        fn('| ')
        for ln in str(q).splitlines():
            fn('| ' + ln)

    q = s.get('expectedResult', '')
    if q and q != 'NOT_APPLICABLE':
        fn('| ')
        fn('| EXPECTED=' + str(q))

    fn('| ')


def _parse_test_results(res):
    all_messages = {}

    for msg in jq.all('.. | .LangTranslationTemplateCollection? | .[]?', res):
        all_messages[msg['name']] = jq.first('.translationTemplates.TranslationTemplate["$"]', msg)
    # the default is just silly
    all_messages['TR.fallbackInfo'] = '{INFO}'

    all_results = []

    for step in _get_list('.. | .TestStepResult?', res):
        if 'testAssertionResults' in step:
            all_results.extend(_get_list('.. | .TestAssertionResult?', step))
        else:
            all_results.append(step)

    return [
        _parse_single_result(q, res, all_messages)
        for q in all_results
    ]


def _parse_single_result(q, res, all_messages):
    r = dict(
        id=q['id'],
        status='passed' if q['status'].startswith('PASSED') else q['status'],
        message='',
        specs=[],
    )

    for m in _get_list('.messages.message', q):
        msg = all_messages.get(m['ref'], '')
        for a in _get_list('.. | .argument?', m):
            msg = msg.replace('{' + a['token'] + '}', str(a['$']))
        r['message'] += m['ref'] + ': ' + msg

    ref = jq.first('. | .resultedFrom? | .href', q)
    if ref:
        r['specs'] = [dict(label=ref)]

    ref = jq.first('. | .resultedFrom? | .ref', q)
    while ref:
        q = jq.first(f'.. | select(.id? == "{ref}")', res)
        r['specs'].append(_pick(q, 'id', 'label', 'description', 'statementForExecution', 'expression', 'expectedResult'))
        ref = jq.first('. | .parent? | .ref', q)

    return r


def _invoke(path, **kwargs):
    url = f'http://{OPTIONS["host"]}:{OPTIONS["port"]}/validator/v2/{path}'
    cli.info(f'>> {url}')
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
    o = {}
    for k in keys:
        if k not in d or d[k] is None:
            o[k] = ''
        else:
            o[k] = d[k]
    return o


def _unhtml(s):
    s = re.sub(r'<[^>]+>', '', s)
    return re.sub(r'\s+', ' ', s.strip())


def _pr(r):
    print(jsonx.to_pretty_string(r))


##

if __name__ == '__main__':
    cli.main('test', main, USAGE)
