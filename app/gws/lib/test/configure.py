import os
import shutil
import yaml
import gws
import gws.lib.test.util as u
import gws.lib.test.services


def empty_dir(wd):
    for de in os.scandir(wd):
        if de.is_dir():
            shutil.rmtree(de.path)
        else:
            os.unlink(de.path)


def pg_service_conf():
    name = u.option('service.postgres.name')
    ini = {
        f'{name}.host': u.option('runner.hostname'),
        f'{name}.port': u.option('service.postgres.port'),
        f'{name}.user': u.option('service.postgres.user'),
        f'{name}.password': u.option('service.postgres.password'),
        f'{name}.dbname': u.option('service.postgres.database'),
    }
    return u.make_ini(ini)


def pytest_ini():
    ini = {
        'pytest.python_files': u.TEST_FILE_GLOB,
        'pytest.cache_dir': u.work_dir() + '/pytest_cache'
    }
    for k, v in u.options().items():
        if k.startswith('pytest.'):
            ini[k] = v
    return u.make_ini(ini)


def compose_yml():
    wd = u.work_dir()

    services = {}

    for s in u.option('runner.services').split(','):
        srv = getattr(gws.lib.test.services, f'configure_{s}')()

        srv.setdefault('image', u.option(f'service.{s}.image'))
        srv.setdefault('extra_hosts', []).append(f"{u.option('runner.hostname')}:host-gateway")
        srv.setdefault('container_name', u.option(f'service.{s}.name'))
        srv.setdefault('volumes', []).append(f"{wd}:{wd}")
        srv.setdefault('tmpfs', []).append('/tmp')
        srv.setdefault('stop_grace_period', '1s')
        srv.setdefault('environment', {})


        for k, v in u.option('environment').items():
            srv['environment'].setdefault(k, v)

        services[s] = srv

    cfg = {
        'version': '3',
        'networks': {
            'default': {
                'name': 'gws_test_network'
            }
        },
        'services': services,
    }

    return yaml.dump(cfg)
