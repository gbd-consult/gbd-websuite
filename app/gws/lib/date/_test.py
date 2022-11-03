import gws.lib.date
import datetime

def test_set_system_time_zone():
    def cat(fn):
        with open(fn, 'rb') as fp:
            return fp.read()

    gws.lib.date.set_system_time_zone('America/Detroit')
    assert cat('/usr/share/zoneinfo/America/Detroit') == cat('/etc/localtime')
    gws.lib.date.set_system_time_zone('Europe/Berlin')
    assert cat('/usr/share/zoneinfo/Europe/Berlin') == cat('/etc/localtime')


def test_to_iso():
    """
    def to_iso(d: datetime.datetime, with_tz='+', sep='T') -> str:

    TODO consider removing the sep='T' seperator option to enforce compliance 
        with iso standard. more information here: 
        https://stackoverflow.com/questions/9531524/in-an-iso-8601-date-is-the-t-character-mandatory
        seperator should always be 'T'. before revision 'T' could be omitted, 
            but never replaced with a ' ' (space)
    
    TODO consider supporting basic and extended formats: 
        basic: 20221021T121212Z or 20221021T121212+0000
        extended: 2022-10-21T12:12:12+0000 or 2022-10-21T12:12:12Z
    
    TODO consider with_tz as flag to enable or disable shortening of +0000 to Z 
        for zero meridian time/gmt. example: def to_iso(d, short_gmt=False)

    NOTE pythons datetime.datetime.fromisoformat(date_string) is not correctly 
        handling all cases, especially the Z suffix or the +0000 without : for +hh:mm
        those are valid in iso, but fail in python builtin

    """
    dt = datetime.datetime.fromisoformat('2022-10-31T16:42:22+09:00')
    assert gws.lib.date.to_iso(dt) == "2022-10-31T16:42:22+0900"
    assert gws.lib.date.to_iso(dt, sep='#') == "2022-10-31#16:42:22+0900"
    dt = datetime.datetime.fromisoformat('2022-10-31T16:42:22+00:00')
    assert gws.lib.date.to_iso(dt, with_tz='+') == "2022-10-31T16:42:22+0000"
    assert gws.lib.date.to_iso(dt, with_tz='Z', sep='#') == "2022-10-31#16:42:22Z"


def test_to_iso_date():
    """
    def to_iso_date(d: datetime.datetime) -> str:
    """

    dt = datetime.datetime.fromisoformat('2022-10-31T16:42:22+09:00')
    assert gws.lib.date.to_iso_date(dt) == "2022-10-31"
    dt = datetime.datetime.fromisoformat('2022-12-31T23:59:59+09:00')
    assert gws.lib.date.to_iso_date(dt) == "2022-12-31"
    dt = datetime.datetime.fromisoformat('2022-12-31T23:59:59-09:00')
    assert gws.lib.date.to_iso_date(dt) == "2022-12-31"


def test_to_iso_local():
    """
    def to_iso_local(d: datetime.datetime, with_tz='+', sep='T') -> str:
    """

    gws.lib.date.set_system_time_zone('Europe/Berlin')

    dt = datetime.datetime.fromisoformat('2022-10-31T16:42:22+09:00')
    assert gws.lib.date.to_iso_local(dt) in [
        "2022-10-31T08:42:22+0100", 
        "2022-10-31T09:42:22+0200"
    ]

    # check for year change on tz difference
    dt = datetime.datetime.fromisoformat('2022-12-31T23:59:59+09:00')
    assert gws.lib.date.to_iso_local(dt) in [
        "2022-12-31T15:59:59+0100", 
        "2022-12-31T16:59:59+0200"
    ] 
    dt = datetime.datetime.fromisoformat('2022-12-31T23:59:59-09:00')
    assert gws.lib.date.to_iso_local(dt) in [
        "2023-01-01T09:59:59+0100", 
        "2023-01-01T10:59:59+0200"
    ]


"""
Not sure how to test now()

def test_now():
    pass

def test_now_iso():
    pass

def test_timestamp():
    pass

def test_utime():
    pass

def test_timestamp_msec():
    pass
"""


def test_to_utc():
    dt = datetime.datetime.fromisoformat("2022-12-12T13:12:12+01:00")
    dt_with_tz = datetime.datetime.fromisoformat("2022-12-12T12:12:12+00:00")
    assert dt.astimezone(datetime.timezone.utc) == dt_with_tz


def test_from_timestamp():
    dt = datetime.datetime.now(tz=datetime.timezone.utc)
    ts = dt.timestamp()
    assert gws.lib.date.from_timestamp(ts) == dt


def test_is_date():
    assert gws.lib.date.is_date(datetime.date.today())
    assert not gws.lib.date.is_date("not date")


def test_is_datetime():
    assert gws.lib.date.is_datetime(datetime.datetime.now())
    assert not gws.lib.date.is_datetime("not datetime")


def test_parse():
    """
    def parse(s):
    """

    assert gws.lib.date.parse(0) == None

    dt = datetime.datetime.fromisoformat('2022-10-31T16:42:22+00:00')
    assert gws.lib.date.parse(dt) == dt

    assert gws.lib.date.parse(20221212) == None
    assert gws.lib.date.parse("20221212") == None

    dt = datetime.datetime.fromisoformat('2022-10-31T16:42:22+00:00')
    assert not gws.lib.date.parse("2022-10-31") == dt

    # these two should definitely have the same result
    dt = datetime.datetime.fromisoformat('2022-10-31T00:00:00+00:00')
    assert gws.lib.date.parse("2022-10-31") == dt
    assert gws.lib.date.parse("31.10.2022") == dt


def test_from_dmy():
    """
    def from_dmy(s: str) -> t.Optional[datetime.datetime]:
    """

    dt = datetime.datetime.fromisoformat('2022-12-31T00:00:00+00:00')
    assert gws.lib.date.from_dmy("31.12.2022") == dt


def test_from_iso():
    """
    def from_iso(s: str) -> t.Optional[datetime.datetime]:
    """

    assert gws.lib.date.from_iso("2022-12-12") == datetime.datetime.fromisoformat("2022-12-12T00:00:00+00:00")
    assert gws.lib.date.from_iso("2022-12-12T00:00:00Z") == datetime.datetime.fromisoformat("2022-12-12T00:00:00+00:00")
    assert gws.lib.date.from_iso("2022-12-12T00:00:00+0000") == datetime.datetime.fromisoformat("2022-12-12T00:00:00+00:00")


def test_date_formatter():
    dfmt = gws.lib.date.date_formatter("de_DE")
    assert dfmt and type(dfmt) == gws.lib.date.DateFormatter

    dt = datetime.datetime.fromisoformat("2022-12-13T14:15:16+00:00")

    dfmt = gws.lib.date.date_formatter("de_DE")
    assert dfmt.format('short',dt) == "13.12.22"
    assert dfmt.format('medium',dt) == "13.12.2022"
    assert dfmt.format('long',dt) == "13. Dezember 2022"

    dfmt = gws.lib.date.date_formatter("en_US")
    assert dfmt.format('short',dt) == "12/13/22"
    assert dfmt.format('medium',dt) == "Dec 13, 2022"
    assert dfmt.format('long',dt) == "December 13, 2022"

    dfmt = gws.lib.date.date_formatter("en_GB")
    assert dfmt.format('short',dt) == "13/12/2022"
    assert dfmt.format('medium',dt) == "13 Dec 2022"
    assert dfmt.format('long',dt) == "13 December 2022"


def test_time_formatter():
    """
    TODO gws.lib.date.TimeFormatter.format() behavior differs from DateFormatter.format()

    """
    tfmt = gws.lib.date.time_formatter("de_DE")
    assert tfmt and type(tfmt) == gws.lib.date.TimeFormatter


    tm = "13:14:15"

    tmft = gws.lib.date.time_formatter("de_DE")
    assert tmft.format('short',tm) == "13:14"
    assert tmft.format('medium',tm) == "13:14:15"
    assert tmft.format('long',tm) == "13:14:15 UTC"

    tmft = gws.lib.date.time_formatter("en_US")
    assert tmft.format('short',tm) == "1:14 PM"
    assert tmft.format('medium',tm) == "1:14:15 PM"
    assert tmft.format('long',tm) == "1:14:15 PM UTC"

    tmft = gws.lib.date.time_formatter("en_GB")
    assert tmft.format('short',tm) == "13:14"
    assert tmft.format('medium',tm) == "13:14:15"
    assert tmft.format('long',tm) == "13:14:15 UTC"

    # this should work because the date equivalence works with DateFormatter
    tm = datetime.time.fromisoformat("13:14:15")
    tmft = gws.lib.date.time_formatter("de_DE")
    #assert tmft.format('short',tm) == "13:14:15"
    assert "INCONSISTENT BEHAVIOR BETWEEN DateFormatter and TimeFormatter: READ COMMENT" == False # fail test manually to continue testing 