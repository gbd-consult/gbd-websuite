"""Utilities for EASD AarteLink push notifications"""

import hashlib

import gws
import gws.tools.misc
import gws.tools.date
import gws.tools.net
import gws.tools.json2
import gws.common.db
import gws.gis.shape

_DEVICE_STATE_VERB = 'receiveDeviceState'
_ALARM_MESSAGE_VERB = 'receiveAlarmMessage'


def service_request(action):
    url = action.var('aarteLink.serviceUrl') + '/' + 'overview'
    auth = (action.var('aarteLink.serviceLogin'), action.var('aarteLink.servicePassword'))
    resp = gws.tools.net.http_request(url, auth=auth)
    r = gws.tools.json2.from_string(resp.text)

    # "version": 1,
    # "timestamp": 1561541113,
    # "time": "2019-06-26T11:25:13+02:00",
    # "messageType": "system_overview",
    # "data": {
    #     "version": 1,
    #     "customerId": ...,
    #     "systemId": "...",
    #     "name": "...",
    #     "comAddr": "...",
    #     "devices": {
    #         "...": {
    #             "name": ...,
    #             "serial": "...",
    #             "state": "...",
    #             "errorLevel": 0,
    #             "errorLevelName": "...",
    #             "type": "...",
    #             "typeName": "...",
    #             "location": "...",

    recs = []

    for id, dev in r['data']['devices'].items():
        rec = {
            'id': id,
            'name': dev.get('name', ''),
            'state': dev.get('state', ''),
            'errorlevel': int(dev.get('errorLevel', 0)),
            'errorlevelname': dev.get('errorLevelName', ''),
            'type': dev.get('type', ''),
            'typename': dev.get('typeName', ''),
        }
        if dev.get('location'):
            x, y = dev.get('location').split(':')
            rec['geom'] = gws.gis.shape.from_geometry({
                'type': 'Point',
                'coordinates': [float(x), float(y)]
            }, 'EPSG:4326')

        recs.append(rec)

    with action.db.connect() as conn:
        conn.exec(f'TRUNCATE TABLE {action.DEVICE_TABLE_NAME}')

    tbl = gws.common.db.SqlTableConfig({
        'name': action.DEVICE_TABLE_NAME,
        'keyColumn': 'id',
        'geometryColumn': 'geom'
    })

    action.db.insert(tbl, recs)


def handle(action, req):
    # because of the encoding issues, we use the raw request_uri and parse it manually
    uri = req.env('REQUEST_URI', '')
    p = uri.split('/')

    for n, s in enumerate(p):
        if s == _DEVICE_STATE_VERB:
            r = _parse_device_state(action, p[n:])
            _save_device_state(action, r)
            return

        if s == _ALARM_MESSAGE_VERB:
            r = _parse_alarm_message(action, p[n:])
            _save_alarm_message(action, r)
            return

    raise ValueError(f'unknown verb: {uri!r}')


def _parse_device_state(action, p):
    # 4.2.2 the url format is
    # <BaseAddress>/receiveDeviceState/<custormerId>/<systemId>/<deviceId>/<values>/<timestamp>/< checksum>

    r = {
        'cmd': p[0],
        'customer_id': p[1],
        'system_id': p[2],
        'device_id': p[3],
        'payload': p[4],
        'values': _parse_device_state_values(p[4]),
        'timestamp': p[5],
        'checksum': p[6],
    }

    # <systemKey><custormerId><systemId><deviceId><values>
    _validate_checksum(action, r, 'customer_id', 'system_id', 'device_id', 'payload')

    return r


def _parse_device_state_values(payload):
    vs = []

    for kv in gws.tools.net.unquote(payload).split(','):
        kv = kv.strip()
        if not kv:
            continue
        m = kv.split('=')
        if len(m) == 2:
            k, v = m
            u = ''
        elif len(m) == 3:
            k, v, u = m
        else:
            raise ValueError(f'unexpected key-value pair {kv!r}')

        # @TODO convert types

        vs.append({
            'name': k.strip(),
            'value': v.strip(),
            'unit': u.strip(),
        })

    return vs


def _parse_alarm_message(action, p):
    # spec 4.2.3
    # <BaseAddress>/receiveAlarmMessage/<custormerId>/<systemId>/<type>/<message>/<timestamp>/<checksum>

    r = {
        'cmd': p[0],
        'customer_id': p[1],
        'system_id': p[2],
        'type': p[3],
        'payload': p[4],
        'message': gws.tools.net.unquote(p[4]),
        'timestamp': p[5],
        'checksum': p[6],
    }

    # md5($systemKey.$customerId.$systemId.$type.$messageEncoded.$timestamp)
    _validate_checksum(action, r, 'customer_id', 'system_id', 'type', 'payload', 'timestamp')

    return r


def _save_device_state(action, r):
    data = [
        gws.extend(v, {
            'customer_id': r['customer_id'],
            'system_id': r['system_id'],
            'device_id': r['device_id'],
            'time_created': _to_date(r),
        })
        for v in r['values']
    ]

    with action.db.connect() as conn:
        conn.batch_insert(action.MESSAGE_TABLE_NAME, data)


def _save_alarm_message(action, r):
    rec = {
        'customer_id': r['customer_id'],
        'system_id': r['system_id'],
        'type': r['type'],
        'message': r['message'],
        'time_created': _to_date(r),
    }

    with action.db.connect() as conn:
        conn.insert_one(action.ALARM_TABLE_NAME, 'id', rec)


def _validate_checksum(action, r, *keys):
    system_key = action.var('aarteLink.systemKey', default='')

    h = system_key + ''.join(r[k] for k in keys)

    md5 = hashlib.md5(h).hexdigest()
    if md5 != r['checksum']:
        gws.log.warn(f"checksum mismatch: h={h!r} cs={r['checksum']!r}")
        # raise ValueError(f"checksum mismatch: h={h!r} cs={r['checksum']!r}")


def _to_date(r):
    return gws.tools.date.to_iso(
        gws.tools.date.from_timestamp(
            int(r['timestamp'])))
