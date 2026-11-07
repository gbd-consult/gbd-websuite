"""Trivy scanner for GWS images."""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../app'))

import gws.lib.cli as cli

USAGE = """
GWS Image Audit
~~~~~~~~~~~~~~~

    python3 audit.py <command> <image-name> <options>

Command:

    scan    - scan an image for security problems
    sbom    - generate an SPDX json report for an image

Options:

    -trivydir <dir>
        directory to store Trivy cache, defaults to $GWS_TRIVY_DIR

    -save-json <path>
        write the scan/sbom report as json to a file

    -save-text <path>
        write the scan report as text to a file

Examples:

    python3 audit.py scan gbdconsult/gws-amd64:8.5.8
    python3 audit.py sbom gbdconsult/gws-amd64:8.5.8 -save-json sbom.spdx.json
"""


class Scanner:
    TRIVY_IMAGE = 'aquasec/trivy'
    SEVERITY = 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'UNKNOWN'

    def __init__(self, cmd, args):
        self.cmd = cmd
        self.args = args

        self.image_name = args.get(2)
        if not self.image_name:
            raise ValueError('image name not set')

        self.trivy_dir = args.get('trivydir') or os.getenv('GWS_TRIVY_DIR') or ''
        if not self.trivy_dir:
            raise ValueError('trivydir not set')

    def main(self):
        if self.cmd == 'scan':
            cli.info(f'SCANNING {self.image_name}')
            return self.do_scan()

        if self.cmd == 'sbom':
            cli.info(f'GENERATING SBOM FOR {self.image_name}')
            return self.do_sbom()

    def do_scan(self):
        js = self.run_trivy(self.image_name, 'json')
        text = self.format_scan_results(json.loads(js))
        
        do_print = True
        p = self.args.get('save-json')
        if p:
            cli.write_file(p, js)
            do_print = False
        p = self.args.get('save-text')
        if p:
            cli.write_file(p, text)
            do_print = False

        if do_print:
            for ln in text.split('\n'):
                if ln.startswith('CRITICAL'):
                    cli.error(ln)
                elif ln.startswith('HIGH'):
                    cli.warning(ln)
                else:
                    cli.info(ln)

        return 0

    def format_scan_results(self, data):
        seen = set()
        rows = []

        for res in data.get('Results', []):
            typ = res.get('Type')
            src_names = {}
            for p in res.get('Packages', []):
                src_names[p.get('Name')] = p.get('SrcName') or p.get('Name')

            for v in res.get('Vulnerabilities', []):
                vid = v.get('VulnerabilityID', '-')
                pkg = v.get('PkgName', '-')
                key = typ, src_names.get(pkg, pkg), vid
                if key in seen:
                    continue
                seen.add(key)

                sev = v.get('Severity', 'UNKNOWN')
                ver = v.get('InstalledVersion', '')
                fix = v.get('FixedVersion', '')
                if fix:
                    ver += f' -> {fix}'
                rows.append(
                    [
                        self.SEVERITY.index(sev),
                        sev + (' (fixed)' if fix else ''),
                        f'{pkg} ({typ})',
                        vid,
                        f'{v.get("Title", "-")} ({ver})',
                    ]
                )

        header = ['Severity', 'Package', 'Vulnerability', 'Title']
        tab = cli.text_table([dict(zip(header, r[1:])) for r in sorted(rows)], header=header)

        head = '\n'.join(
            [
                'CVE REPORT\n',
                'CreatedAt:        ' + data['CreatedAt'],
                'Metadata.ImageID: ' + data['Metadata']['ImageID'],
                'ArtifactName:     ' + data['ArtifactName'],
            ]
        )

        return head + '\n\n' + tab + '\n'

    def do_sbom(self):
        text = self.run_trivy(self.image_name, 'spdx-json')

        path = self.args.get('save-json')
        if path:
            cli.write_file(path, text)
            return 0

        print(text)
        return 0

    def run_trivy(self, image_name, report_format):

        if not os.path.isdir(self.trivy_dir):
            os.makedirs(self.trivy_dir, exist_ok=True)

        # ignore linux-libc-dev: kernel CVEs, containers run on the host's kernel
        _rego_policy = """
            package trivy
            default ignore = false
            ignore {
                input.PkgName == "linux-libc-dev"
            }
        """
        cli.write_file(f'{self.trivy_dir}/rego_policy.rego', _rego_policy)

        out = '__out.json'
        try:
            os.unlink(f'{self.trivy_dir}/{out}')
        except OSError:
            pass

        cmd = f"""
            docker run
            --mount type=bind,src={self.trivy_dir},dst=/root/.cache/
            --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock
            {self.TRIVY_IMAGE}
            image
            --format {report_format}
            --scanners vuln
            --output /root/.cache/{out}
            --ignore-policy /root/.cache/rego_policy.rego
            {image_name}
        """

        cli.run(cmd)

        return cli.read_file(f'{self.trivy_dir}/{out}')


###


def main(args):
    cmd = args.get(1)
    if not cmd:
        cli.fatal(f'command required')
    if cmd in {'scan', 'sbom'}:
        return Scanner(cmd, args).main()
    cli.fatal(f'invalid command {cmd!r}')


if __name__ == '__main__':
    cli.main('audit.py', main, USAGE)
