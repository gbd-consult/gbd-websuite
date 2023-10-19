"""Test services."""

import os
import gws
import gws.lib.test.util as u


def configure_gws():
    wd = u.work_dir()

    os.mkdir(f'{wd}/gws-var')
    os.mkdir(f'{wd}/gws-tmp')

    return {
        'ports': [
            f"{u.option('service.gws.http_port')}:80",
            f"{u.option('service.gws.mpx_port')}:5000",
        ],
        # 'command': 'sleep infinity',
        'volumes': [
            f"{u.APP_DIR}:/gws-app",
            f"{wd}/gws-var:/gws-var",
            # f"{wd}/gws-tmp:/tmp",
            f"{u.option('service.gws.datadir')}:/data",
        ],
    }


def configure_postgres():
    # https://github.com/docker-library/docs/blob/master/postgres/README.md

    return {
        'environment': {
            'POSTGRES_DB': u.option('service.postgres.database'),
            'POSTGRES_PASSWORD': u.option('service.postgres.password'),
            'POSTGRES_USER': u.option('service.postgres.user'),
        },
        'ports': [
            f"{u.option('service.postgres.port')}:5432",
        ],
        'volumes': [
            f"{u.option('service.postgres.datadir')}:/var/lib/postgresql/data",
        ],
    }


def configure_mockserv():
    wd = u.work_dir()
    server_app = 'mockserv.app.py'
    reqs_file = 'mockserv_requirements.txt'

    gws.write_file(
        f"{wd}/{server_app}",
        gws.read_file(f'{u.APP_DIR}/gws/lib/test/{server_app}')
    )

    gws.write_file(
        f"{wd}/{reqs_file}",
        '\n'.join(u.option('service.mockserv.requirements').split()) + '\n'
    )

    command = f"bash -c 'pip install -r {wd}/{reqs_file} && python {wd}/{server_app}'"

    return {
        'command': command,
        'ports': [
            f"{u.option('service.mockserv.port')}:8080",
        ],
    }


def configure_qgis():
    command = f"/bin/sh /qgis-start.sh"

    return {
        'command': command,
        'ports': [
            f"{u.option('service.qgis.port')}:80",
        ],
    }
