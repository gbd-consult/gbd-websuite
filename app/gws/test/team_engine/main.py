"""

https://opengeospatial.github.io/teamengine/users.html
https://github.com/opengeospatial/teamengine

"""

import os
import sys
import re
import base64
import html
import shutil

LOCAL_APP_DIR = os.path.abspath(os.path.dirname(__file__) + '/../../..')
sys.path.insert(0, LOCAL_APP_DIR)

import gws
import gws.lib.cli as cli
import gws.lib.net
import gws.lib.xmlx as xmlx

USERNAME = "gwstest"
PASSWORD = "gws"

USER_DIR = os.path.dirname(__file__) + f'/te_base/users/{USERNAME}'

USAGE = """
GWS TeamEngine runner
~~~~~~~~~~~~~~~~~~~~~

Commands:

    main.py ls
        - show available test suites

    main.py args <suite>
        - show arguments for the given suite

    main.py run <suite> <url>
        - run a suite against the given URL

Options:
    --host                - TeamEngine host (default: 127.0.0.1)
    --port                - TeamEngine port (default: 8090)
    --level               - list of error levels to report (default: 4,6) or 'all'
    
    --save-xml <path>     - save raw XML output
    
The TE docker container must be started before running this script (see `./docker-compose.yml`).
    
"""

OPTIONS = {}

# https://www.w3.org/TR/EARL10-Schema/#OutcomeValue + http://cite.opengeospatial.org/earl#inheritedFailure"
# using CTL XML error levels from https://opengeospatial.github.io/teamengine/users.html

ERROR_LEVELS = {
    'passed': ('pass', 1),
    'inapplicable': ('pass', 2),
    'untested': ('warn', 3),
    'cantTell': ('warn', 4),
    'inheritedFailure': ('FAIL', 5),
    'failed': ('FAIL', 6),
}

STATUS_MARKS = {
    'FAIL': '\u274C',
    'warn': '\u2754',
    'pass': '\u2705',
}


def main(args):
    OPTIONS['host'] = args.get('host', 'localhost')
    OPTIONS['port'] = args.get('port', '8080')
    OPTIONS['save-xml'] = args.get('save-xml', '')

    s = args.get('level', '4,6')
    if s == 'all':
        OPTIONS['level'] = list(range(7))
    else:
        OPTIONS['level'] = [int(x) for x in gws.u.to_list(s)]

    cmd = args.get(1)

    if cmd == 'ls':
        print(cli.text_table(_get_suites(), header='auto'))
        return 0

    if cmd == 'args':
        suite = args.get(2)
        print(cli.text_table(_get_args(suite)))
        return 0

    if cmd == 'run':
        shutil.rmtree(f'{USER_DIR}/rest', ignore_errors=True)
        OPTIONS['suite'] = args.get(2)
        OPTIONS['url'] = args.get(3)
        xml = _invoke_test()
        results = _parse_test_results(xml)
        _report_test(results)
        return 0


def _get_suites():
    xml = xmlx.from_string(_invoke('suites'))
    return sorted(
        [el.textdict() for el in xml.findall('.//testSuite')],
        key=str
    )


def _get_args(suite):
    xml = xmlx.from_string(_invoke(f'suites/{suite}'))
    return [el.textdict() for el in xml.findall('.//testrunargument')]


URL_ARG = {
    'wfs20': 'wfs',
    'wms13': 'capabilities-url'
}


def _invoke_test():
    if not OPTIONS['url'].startswith('http'):
        OPTIONS['url'] = 'http://' + OPTIONS['url']
    url = OPTIONS['url']
    if '?' not in url:
        url += '?request=GetCapabilities'
    url = url.replace('&', '&amp;')

    suite = OPTIONS['suite']
    text = _invoke(f'suites/{suite}/run?{URL_ARG[suite]}={url}', accept='application/rdf+xml')
    if OPTIONS['save-xml']:
        gws.u.write_file(OPTIONS['save-xml'], text)

    return xmlx.from_string(text, remove_namespaces=True)


def _parse_test_results(xml):
    # parse test results in the EARL format (https://www.w3.org/TR/EARL10-Guide)

    tc_map = {}
    results = []

    for case_el in xml.findall('.//TestCase'):
        uid = case_el.get('about')
        tc = dict(
            testCase=uid,
            title=_nows(case_el.textof('title')),
            description=_nows(html.unescape(case_el.textof('description'))),
            details='',
        )
        if re.match(r'^s\d+', uid):
            # uids starting with sNNNN reference log files in the rest dir
            path = USER_DIR + '/rest/' + uid.split('#')[0] + '/log.xml'
            tc['details'] = _parse_ctl_log(path)

        tc_map[uid] = tc

    for ass_el in xml.findall('.//Assertion'):
        test_el = ass_el.find('test')

        # it's either an inline <test><TestCase about=ID> or a reference <test resource=ID>
        case_el = test_el.find('TestCase')
        tc = tc_map[case_el.get('about')] if case_el else tc_map[test_el.get('resource')]

        res_el = ass_el.find('result/TestResult')

        r = dict(tc)
        # eg <earl:outcome rdf:resource="http://www.w3.org/ns/earl#passed"/>
        r['status2'] = res_el.find('outcome').get('resource').split('#')[-1]
        r['status'], r['level'] = ERROR_LEVELS[r['status2']]

        # eg <earl:Assertion rdf:about="assert-7">
        r['uid'] = int(re.sub(r'\D+', '', ass_el.get('about')))

        s = res_el.textof('description')
        if s and s != 'No details available.':
            r['details'] = s + '\n' + r['details']

        results.append(r)

    return sorted(results, key=lambda r: r['uid'])

def _parse_ctl_log(path):
    if not os.path.exists(path):
        return ''

    xml = xmlx.from_path(path, remove_namespaces=True)

    desc = [el.text for el in xml.findall('.//message')]

    params = []
    for el in xml.findall('.//param'):
        if el.get('name'):
            params.append(el.get('name', '') + '=' + (el.text or ''))
    if params:
        desc.append(OPTIONS['url'].split('?')[0] + '?' + '&'.join(params))

    return '\n'.join(desc)


def _report_test(results):
    stats = f'TOTAL={len(results)}'

    by_status = {}
    for r in results:
        by_status[r['status2']] = by_status.get(r['status2'], 0) + 1
    for k, v in sorted(by_status.items()):
        stats += f' {k}={v}'

    cli.info(f'')
    cli.info(stats)
    cli.info(f'')

    results = [r for r in results if r['level'] in OPTIONS['level']]

    for r in results:
        print(_nows(f"""
            {STATUS_MARKS[r['status']]}  
            {r['status']} 
            ({r['status2']}, {r['level']}): 
            {r['title']}. {r['description']}
            [{r['testCase']}]
        """))

        if r['details']:
            for ln in r['details'].splitlines():
                print(f'   | {ln}')

        print()

    return


def _invoke(path, **kwargs):
    url = f'http://{OPTIONS["host"]}:{OPTIONS["port"]}/teamengine/rest/{path}'
    headers = {
        'Authorization': f'Basic ' + base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode(),
        'Accept': kwargs.pop('accept', 'application/xml'),
    }
    cli.info(f'>> {url}')
    try:
        res = gws.lib.net.http_request(url, headers=headers, **kwargs)
        if res.status_code != 200:
            cli.error(res.text)
            cli.fatal(f'HTTP ERROR {res.status_code}')
        return res.text
    except gws.lib.net.Error as exc:
        cli.fatal(f'HTTP ERROR: {exc!r}')


def _nows(s):
    return re.sub(r'\s+', ' ', s.strip())


if __name__ == '__main__':
    cli.main('test', main, USAGE)
