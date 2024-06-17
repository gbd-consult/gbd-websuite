import datetime as dt

import gws
import gws.lib.datetimex as dx
import gws.test.util as u


def test_set_local_time_zone():
    if not gws.env.GWS_IN_CONTAINER:
        # needs to be root
        return

    dat = '2001-02-03T11:22:33'
    cmd = f'date --date="{dat}" --iso-8601="seconds"'

    # preserve the configured tz
    orig_dt = u.exec(cmd)
    orig_tz = u.exec('readlink /etc/localtime')

    dx.set_local_time_zone('America/Detroit')
    assert u.exec(cmd) == f'{dat}-05:00'  # EST (UTC-5)

    dx.set_local_time_zone('Asia/Kamchatka')
    assert u.exec(cmd) == f'{dat}+12:00'  # PETT (UTC+12)

    # restore
    u.exec(f'ln -fs {orig_tz} /etc/localtime')
    assert u.exec(cmd) == orig_dt


def test_time_zone():
    s = 'Europe/Paris'
    tz = dx.time_zone(s)
    assert str(tz) == s


def test_new():
    a = dx.new(2024, 1, 2, 3, 4, 5)
    b = dt.datetime(2024, 1, 2, 3, 4, 5).astimezone()
    assert a.isoformat() == b.isoformat()

    a = dx.new(2024, 1, 2, 3, 4, 5, tz='utc')
    b = dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)
    assert a.isoformat() == b.isoformat()


def test_now():
    a = dx.now()
    b = dt.datetime.now().astimezone()
    assert (b - a).total_seconds() < 2


def test_parse():
    s = '2000-01-02T03:04:05'
    a = 2000, 1, 2, 3, 4, 5

    # explicit tz

    d = dx.parse('2000-01-02T03:04:05' + '+0000')
    e = dt.datetime(*a, tzinfo=dx.time_zone('utc'))
    assert d.isoformat() == e.isoformat()

    d = dx.parse('2000-01-02T03:04:05' + 'Z')
    e = dt.datetime(*a, tzinfo=dx.time_zone('utc'))
    assert d.isoformat() == e.isoformat()

    d = dx.parse(s + '+0900')
    # NB, the sign is inverted, see https://github.com/eggert/tz/blob/2018g/etcetera#L38
    e = dt.datetime(*a, tzinfo=dx.time_zone('Etc/GMT-9'))
    assert d.isoformat() == e.isoformat()

    # default tz

    d = dx.parse(s, tz='Etc/GMT-5')
    e = dt.datetime(*a, tzinfo=dx.time_zone('Etc/GMT-5'))
    assert d.isoformat() == e.isoformat()

    # local tz

    d = dx.parse(s)
    e = dt.datetime(*a, tzinfo=dx.time_zone(''))
    assert d.isoformat() == e.isoformat()

    # date only

    d = dx.parse('2000-01-02')
    e = dt.datetime(2000, 1, 2, tzinfo=dx.time_zone(''))
    assert d.isoformat() == e.isoformat()

    d = dx.parse('2000-01-02', tz='Etc/GMT-5')
    e = dt.datetime(2000, 1, 2, tzinfo=dx.time_zone('Etc/GMT-5'))
    assert d.isoformat() == e.isoformat()

    # year only

    d = dx.parse('2000')
    e = dt.datetime(2000, 1, 1, tzinfo=dx.time_zone(''))
    assert d.isoformat() == e.isoformat()

    # not supported

    d = dx.parse('10:11:12')
    assert d is None

    d = dx.parse('2007-03-01T13:00:00Z/2008-05-11T15:30:00Z')
    assert d is None


def test_parse_duration():
    assert dx.parse_duration('1 w') == 3600 * 24 * 7
    assert dx.parse_duration('1 d') == 3600 * 24
    assert dx.parse_duration('1 h') == 3600
    assert dx.parse_duration('1 m') == 60
    assert dx.parse_duration('1 s') == 1


def test_parse_duration_errors():
    with u.raises(dx.Error, match=f'invalid duration'):
        dx.parse_duration('1 foo')


def test_parse_time():
    with dx.mock_now(dx.new(2023, 3, 26, 0, 0, 0, tz='Europe/Berlin')):
        d = dx.parse_time('01:02:03', tz='Europe/Berlin')
        assert dx.to_iso_string(d) == '2023-03-26T01:02:03+0100'
        d = dx.parse_time('03:04:05', tz='Europe/Berlin')
        assert dx.to_iso_string(d) == '2023-03-26T03:04:05+0200'


def test_to_iso_string():
    d = dt.datetime.fromisoformat('2022-10-31T16:42:22+09:00')
    assert dx.to_iso_string(d) == '2022-10-31T16:42:22+0900'
    assert dx.to_iso_string(d, sep='#') == '2022-10-31#16:42:22+0900'

    d = dt.datetime.fromisoformat('2022-10-31T16:42:22+00:00')
    assert dx.to_iso_string(d, with_tz='+') == '2022-10-31T16:42:22+0000'
    assert dx.to_iso_string(d, with_tz='Z', sep='#') == '2022-10-31#16:42:22Z'


def test_to_iso_date_string():
    d = dt.datetime.fromisoformat('2022-10-31T16:42:22+09:00')
    assert dx.to_iso_date_string(d) == '2022-10-31'
    d = dt.datetime.fromisoformat('2022-12-31T23:59:59+09:00')
    assert dx.to_iso_date_string(d) == '2022-12-31'
    d = dt.datetime.fromisoformat('2022-12-31T23:59:59-09:00')
    assert dx.to_iso_date_string(d) == '2022-12-31'


def test_add():
    a = dt.datetime.fromisoformat('2000-01-30T03:04:05+00:00')
    b = dx.add(a, days=5)
    assert b.isoformat() == '2000-02-04T03:04:05+00:00'
    b = dx.add(a, days=5, hours=-5)
    assert b.isoformat() == '2000-02-03T22:04:05+00:00'


def test_add_over_dst():
    a = dx.new(2023, 3, 26, 0, 11, 22, tz='Europe/Berlin')
    assert a.isoformat() == '2023-03-26T00:11:22+01:00'

    assert dx.add(a, hours=1).isoformat() == '2023-03-26T01:11:22+01:00'
    assert dx.add(a, hours=2).isoformat() == '2023-03-26T02:11:22+01:00'  #
    assert dx.add(a, hours=3).isoformat() == '2023-03-26T03:11:22+02:00'  #
    assert dx.add(a, hours=4).isoformat() == '2023-03-26T04:11:22+02:00'

    t = a.timestamp()

    assert dx.add(a, hours=1).timestamp() == t + 1.0 * 3600
    assert dx.add(a, hours=2).timestamp() == t + 2.0 * 3600  #
    assert dx.add(a, hours=3).timestamp() == t + 2.0 * 3600  #
    assert dx.add(a, hours=4).timestamp() == t + 3.0 * 3600


def _diff(years=0, months=0, weeks=0, days=0, hours=0, minutes=0, seconds=0, microseconds=0):
    return dict(
        years=years or 0,
        months=months or 0,
        weeks=weeks or 0,
        days=days or 0,
        hours=hours or 0,
        minutes=minutes or 0,
        seconds=seconds or 0,
        microseconds=microseconds or 0,
    )


def test_difference():
    h, m, s = 4, 5, 6

    a = dt.datetime.fromisoformat('2000-01-30T03:04:05+00:00')
    b = dx.add(a, seconds=3600 * h + 60 * m + s)
    df = dx.difference(a, b)

    assert vars(df) == _diff(hours=h, minutes=m, seconds=s)


def test_total_difference_hours():
    h, m, s = 4, 5, 6

    a = dt.datetime.fromisoformat('2000-01-30T03:04:05+00:00')
    b = dx.add(a, seconds=3600 * h + 60 * m + s)
    df = dx.total_difference(a, b)

    assert vars(df) == _diff(
        hours=h,
        minutes=h * 60 + m,
        seconds=h * 3600 + m * 60 + s,
        microseconds=(h * 3600 + m * 60 + s) * 1_000_000
    )


def test_total_difference_weeks():
    w = 6

    a = dt.datetime.fromisoformat('2000-01-30T03:04:05+00:00')
    b = dx.add(a, weeks=w)
    df = dx.total_difference(a, b)

    assert vars(df) == _diff(
        months=1,
        weeks=w,
        days=w * 7,
        hours=w * 7 * 24,
        minutes=w * 7 * 24 * 60,
        seconds=w * 7 * 24 * 3600,
        microseconds=(w * 7 * 24 * 3600) * 1_000_000
    )
