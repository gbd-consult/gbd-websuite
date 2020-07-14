import os
import gws
import pwd
import grp

import gws.config
import gws.qgis.server
import gws.tools.os2

import gws.types as t

MAPPROXY_YAML_PATH = gws.CONFIG_DIR + '/mapproxy.yaml'

# https://uwsgi-docs.readthedocs.io/en/latest/Nginx.html

_uwsgi_params = """
    uwsgi_param QUERY_STRING $query_string;
    uwsgi_param REQUEST_METHOD $request_method;
    uwsgi_param CONTENT_TYPE $content_type;
    uwsgi_param CONTENT_LENGTH $content_length;
    uwsgi_param REQUEST_URI $request_uri;
    uwsgi_param PATH_INFO $document_uri;
    uwsgi_param DOCUMENT_ROOT $document_root;
    uwsgi_param SERVER_PROTOCOL $server_protocol;
    uwsgi_param REMOTE_ADDR $remote_addr;
    uwsgi_param REMOTE_PORT $remote_port;
    uwsgi_param SERVER_ADDR $server_addr;
    uwsgi_param SERVER_PORT $server_port;
    uwsgi_param SERVER_NAME $server_name;
"""


def create(root: t.IRootObject, base_dir, pid_dir):
    def _write(p, s):
        p = base_dir + '/' + p
        s = '\n'.join(x.strip() for x in s.strip().splitlines())
        with open(p, 'wt') as fp:
            fp.write(s + '\n')
        return p

    for p in gws.tools.os2.find_files(base_dir, '(conf|ini)$'):
        gws.tools.os2.unlink(p)

    commands = []
    frontends = []

    # Check the marker file created by our docker build (see install/build.py)
    try:
        in_container = os.path.isfile('/.GWS_IN_CONTAINER')
    except:
        in_container = False

    rsyslogd_enabled = in_container

    # NB it should be possible to have QGIS running somewhere else
    # so, if 'host' is not localhost, don't start QGIS here
    qgis_enabled = root.var('server.qgis.enabled') and root.var('server.qgis.host') == 'localhost'
    qgis_port = root.var('server.qgis.port')
    qgis_workers = root.var('server.qgis.workers')
    qgis_threads = root.var('server.qgis.threads')
    qgis_socket = gws.TMP_DIR + '/uwsgi.qgis.sock'

    web_enabled = root.var('server.web.enabled')
    web_workers = root.var('server.web.workers')
    web_threads = root.var('server.web.threads')
    web_socket = gws.TMP_DIR + '/uwsgi.web.sock'

    spool_enabled = root.var('server.spool.enabled')
    spool_workers = root.var('server.spool.workers')
    spool_threads = root.var('server.spool.threads')
    spool_socket = gws.TMP_DIR + '/uwsgi.spooler.sock'
    spool_dir = gws.SPOOL_DIR
    spool_freq = root.var('server.spool.jobFrequency')

    mapproxy_enabled = root.var('server.mapproxy.enabled') and os.path.exists(MAPPROXY_YAML_PATH)
    mapproxy_port = root.var('server.mapproxy.port')
    mapproxy_workers = root.var('server.mapproxy.workers')
    mapproxy_threads = root.var('server.mapproxy.threads')
    mapproxy_socket = gws.TMP_DIR + '/uwsgi.mapproxy.sock'

    log = root.var('server.log') or ('syslog' if in_container else gws.LOG_DIR + '/gws.log')

    nginx_log_level = 'info'
    if root.application.developer_option('nginx.log_level_debug'):
        nginx_log_level = 'debug'

    nginx_rewrite_log = 'off'
    if root.application.developer_option('nginx.rewrite_log_on'):
        nginx_rewrite_log = 'on'

    if log == 'syslog':
        nginx_log = 'syslog:server=unix:/dev/log,nohostname,tag'

        nginx_main_log = f'{nginx_log}=NGINX_MAIN'
        nginx_qgis_log = f'{nginx_log}=NGINX_QGIS'
        nginx_web_log = f'{nginx_log}=NGINX_WEB'

        uwsgi_qgis_log = 'daemonize=true\nlogger=syslog:QGIS,local6'
        uwsgi_web_log = 'daemonize=true\nlogger=syslog:WEB,local6'
        uwsgi_mapproxy_log = 'daemonize=true\nlogger=syslog:MAPPROXY,local6'
        uwsgi_spool_log = 'daemonize=true\nlogger=syslog:SPOOL,local6'

    else:
        nginx_main_log = nginx_qgis_log = nginx_web_log = log
        uwsgi_qgis_log = uwsgi_web_log = uwsgi_mapproxy_log = uwsgi_spool_log = f'daemonize={log}'

    # be rude and reload 'em as fast as possible
    mercy = 5

    # @TODO: do we need more granular timeout configuration?
    qgis_timeout = root.var('server.timeout')
    qgis_front_timeout = qgis_timeout + 10
    mapproxy_timeout = qgis_front_timeout + 10
    web_timeout = mapproxy_timeout + 10
    web_front_timeout = web_timeout + 10
    spool_timeout = 120

    stdenv = '\n'.join(
        f'env = {k}={v}'
        for k, v in os.environ.items()
        if k.startswith('GWS_')
    )

    stdenv += f'\nTMP={gws.TMP_DIR}'
    stdenv += f'\nTEMP={gws.TMP_DIR}'

    # rsyslogd
    # ---------------------------------------------------------

    if rsyslogd_enabled:
        #  based on /etc/rsyslog.conf
        syslog_conf = f"""
            ##
            
            module(
                load="imuxsock"
                SysSock.UsePIDFromSystem="on"
            )
    
            module(
                load="imklog" 
                PermitNonKernelFacility="on"
            )
            
            template(name="gws" type="list") {{
                property(name="timestamp" dateFormat="rfc3339")
                constant(value=" ")
                property(name="syslogtag")
                constant(value=" ")
                property(name="msg" spifno1stsp="on" )
                property(name="msg" droplastlf="on" )
                constant(value="\\n")
            }}
    
            module(
                load="builtin:omfile" 
                Template="gws"
            )
    
    
            # *.*;kern.none /dev/stdout
            # kern.*	      -/var/log/kern.log
            
            *.* /dev/stdout
        """

        path = _write('syslog.conf', syslog_conf)
        commands.append(f'rsyslogd -i {pid_dir}/rsyslogd.pid -f {path}')

    # qgis
    # ---------------------------------------------------------

    if qgis_enabled:

        # partially inspired by
        # https://github.com/elpaso/qgis2-server-vagrant/blob/master/docs/index.rst

        # harakiri doesn't seem to work with worker-exec
        # besides, it's a bad idea anyways, because killing them prematurely
        # doesn't give them a chance to fully preload a project

        srv = gws.qgis.server.EXEC_PATH
        ini = f"""
            [uwsgi]
            uid = {gws.UID}
            gid = {gws.GID}
            chmod-socket = 777
            fastcgi-socket = {qgis_socket}
            {uwsgi_qgis_log}
            master = true
            pidfile = {pid_dir}/qgis.uwsgi.pid
            processes = {qgis_workers}
            reload-mercy = {mercy}
            threads = {qgis_threads}
            vacuum = true
            worker-exec = {srv}
            worker-reload-mercy = {mercy}
            {stdenv}
        """

        for k, v in gws.qgis.server.environ(root).items():
            ini += f'env = {k}={v}\n'

        path = _write('uwsgi_qgis.ini', ini)
        commands.append(f'uwsgi --ini {path}')

        frontends.append(f"""
            server {{
                listen {qgis_port};
            
                server_name qgis;
                error_log {nginx_qgis_log} {nginx_log_level};
                access_log {nginx_qgis_log};
                rewrite_log {nginx_rewrite_log};
            
                location / {{
                    gzip off;
                    fastcgi_pass unix:{qgis_socket};
                    fastcgi_read_timeout {qgis_front_timeout};
                    
                    # add_header 'Access-Control-Allow-Origin' *;
                    # add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range';
                    
                    # replace mapproxy forward params (e.g. LAYERS__gws) with their real names
                    
                    if ($args ~* (.*?)(layers=-)(.*)) {{
                        set $args $1$3;
                    }}
                    if ($args ~ (.*?)(__gws)(.*)) {{
                        set $args $1$3;
                    }}

                    include /etc/nginx/fastcgi_params;
                }}
            }}
        """)

    # web
    # ---------------------------------------------------------

    if web_enabled:
        ini = f"""
            [uwsgi]
            uid = {gws.UID}
            gid = {gws.GID}
            buffer-size = 65535 
            chmod-socket = 777
            die-on-term = true
            harakiri = {web_timeout}
            harakiri-verbose = true
            {uwsgi_web_log}
            pidfile = {pid_dir}/web.uwsgi.pid
            post-buffering = 65535
            processes = {web_workers}
            pythonpath = {gws.APP_DIR}
            reload-mercy = {mercy}
            spooler-external = {spool_dir}   
            threads = {web_threads}
            uwsgi-socket = {web_socket}
            vacuum = true
            worker-reload-mercy = {mercy}
            wsgi-file = {gws.APP_DIR}/gws/web/web.wsgi.py
            {stdenv}
        """

        path = _write('uwsgi_web.ini', ini)
        commands.append(f'uwsgi --ini {path}')

        roots = ''
        rewr = ''

        app = gws.config.root().application
        for site in app.web_sites:
            for r in site.rewrite_rules:
                rewr += f'rewrite {r.match} {r.target} last;\n'

            d = site.static_root.dir
            roots += f"""
                location =/ {{
                    root {d};
                    index index.html;
                }}
                location / {{
                    root {d};
                    try_files $uri @cache;
                }}
            """
            # @TODO multisites
            break

        # this is in MB
        max_body_size = root.var('server.web.maxRequestLength')

        client_buffer_size = 4  # MB
        client_tmp_dir = gws.ensure_dir(gws.TMP_DIR + '/nginx')

        web_common = f"""
            error_log {nginx_web_log} {nginx_log_level};
            access_log {nginx_web_log} apm;
            rewrite_log {nginx_rewrite_log};
        
            client_max_body_size {max_body_size}m;
            client_body_buffer_size {client_buffer_size}m;
            client_body_temp_path {client_tmp_dir};
            
            # @TODO: optimize, disallow _ rewriting

            {rewr}
            {roots}
            
            location /gws-client/ {{
                root {gws.APP_DIR}/www;
                try_files $uri @app;
            }}

            location @cache {{
                root {gws.WEB_CACHE_DIR};
                try_files $uri @app;
            }}

            location @app {{
                uwsgi_pass unix://{web_socket};
                uwsgi_read_timeout {web_front_timeout};
                {_uwsgi_params}
            }}
        """

        ssl_crt = root.var('web.ssl.crt')
        ssl_key = root.var('web.ssl.key')

        ssl_hsts = ''
        s = root.var('web.ssl.hsts')
        if s:
            ssl_hsts = f'add_header Strict-Transport-Security "max-age={s}; includeSubdomains";'

        # NB don't include xml (some WMS clients don't understand gzip)
        # text/xml application/xml application/xml+rss

        gzip = """
            gzip on;
            gzip_types text/plain text/css application/json application/javascript text/javascript;
        
            # gzip_vary on;
            # gzip_proxied any;
            # gzip_comp_level 6;
            # gzip_buffers 16 8k;
            # gzip_http_version 1.1;
        """

        if ssl_crt:
            frontends.append(f"""
                server {{
                    listen 80 default_server;
                    server_name gws;
                    return 301 https://$host$request_uri;
                }}

                server {{
                    listen 443 ssl default_server;
                    server_name gws;
    
                    ssl_certificate     {ssl_crt};
                    ssl_certificate_key {ssl_key};
                    {ssl_hsts}

                    {gzip}
                    {web_common}
                }}
            """)

        else:
            frontends.append(f"""
                server {{
                    listen 80 default_server;
                    server_name gws;
                    {gzip}
                    {web_common}
                    
                }}
        """)

    # mapproxy
    # ---------------------------------------------------------

    # see https://github.com/mapproxy/mapproxy/issues/282 about 'wsgi-disable-file-wrapper'

    if mapproxy_enabled:
        ini = f"""
            [uwsgi]
            uid = {gws.UID}
            gid = {gws.GID}
            chmod-socket = 777
            die-on-term = true
            harakiri = {mapproxy_timeout}
            harakiri-verbose = true
            http = :{mapproxy_port}
            http-to = {mapproxy_socket}
            {uwsgi_mapproxy_log}
            pidfile = {pid_dir}/mapproxy.uwsgi.pid
            post-buffering = 65535
            processes = {mapproxy_workers}
            pythonpath = {gws.APP_DIR}
            reload-mercy = {mercy}
            threads = {mapproxy_threads}
            uwsgi-socket = {mapproxy_socket}
            vacuum = true
            worker-reload-mercy = {mercy}
            wsgi-disable-file-wrapper = true
            wsgi-file = {gws.APP_DIR}/gws/gis/mpx/mpx.wsgi.py
            {stdenv}
        """

        path = _write('uwsgi_mapproxy.ini', ini)
        commands.append(f'uwsgi --ini {path}')

    # spooler
    # ---------------------------------------------------------

    if spool_enabled:
        ini = f"""
            [uwsgi]
            uid = {gws.UID}
            gid = {gws.GID}
            chmod-socket = 777
            die-on-term = true
            harakiri = {spool_timeout}
            harakiri-verbose = true
            {uwsgi_spool_log}
            master = true
            pidfile = {pid_dir}/spool.uwsgi.pid
            post-buffering = 65535
            processes = {spool_workers}
            pythonpath = {gws.APP_DIR}
            reload-mercy = {mercy}
            spooler = {spool_dir}
            spooler-frequency = {spool_freq}
            threads = {spool_threads}
            uwsgi-socket = {spool_socket}
            vacuum = true
            worker-reload-mercy = {mercy}
            wsgi-file = {gws.APP_DIR}/gws/server/spool.wsgi.py
            {stdenv}
        """

        path = _write('uwsgi_spool.ini', ini)
        commands.append(f'uwsgi --ini {path}')

    # main
    # ---------------------------------------------------------

    frontends = '\n\n'.join(frontends)

    # log_format: https://www.nginx.com/blog/using-nginx-logging-for-application-performance-monitoring/

    u = pwd.getpwuid(gws.UID).pw_name
    g = grp.getgrgid(gws.GID).gr_name

    daemon = 'daemon off;' if in_container else ''

    nginx_conf = f"""
        worker_processes auto;
        pid {pid_dir}/nginx.pid;
        user {u} {g};
        
        events {{
            worker_connections 768;
            # multi_accept on;
        }}
        
        {daemon}

        error_log {nginx_main_log} {nginx_log_level};
        
        http {{
            log_format apm '$remote_addr'
                           ' method=$request_method request="$request"'
                           ' request_length=$request_length'
                           ' status=$status bytes_sent=$bytes_sent'
                           ' referer=$http_referer'
                           ' user_agent="$http_user_agent"'
                           ' request_time=$request_time'
                           ' upstream_response_time=$upstream_response_time'
                           ' upstream_connect_time=$upstream_connect_time'
                           ' upstream_header_time=$upstream_header_time';
            

            access_log {nginx_main_log};
        
            sendfile on;
            tcp_nopush on;
            tcp_nodelay on;
            keepalive_timeout 65;
            types_hash_max_size 2048;

            include /etc/nginx/mime.types;
            default_type application/octet-stream;
        
            ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
            ssl_prefer_server_ciphers on;
        
            {frontends}
        }}
    """

    path = _write('nginx.conf', nginx_conf)
    commands.append(f'nginx -c {path}')

    return commands
