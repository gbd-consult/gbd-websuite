import gws.lib.datetimex as datetimex
import gws.lib.intl as intl

de = dict(
    uid='de_DE',
    dateFormatLong='d. MMMM y',
    dateFormatMedium='dd.MM.y',
    dateFormatShort='dd.MM.yy',
    dateUnits='JMT',
    dayNamesLong=['Sonntag', 'Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag'],
    dayNamesNarrow=['S', 'M', 'D', 'M', 'D', 'F', 'S'],
    dayNamesShort=['So.', 'Mo.', 'Di.', 'Mi.', 'Do.', 'Fr.', 'Sa.'],
    firstWeekDay=0,
    language3='deu',
    language='de',
    languageBib='ger',
    languageName='Deutsch',
    languageNameEn='German',
    monthNamesLong=['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember'],
    monthNamesNarrow=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'],
    monthNamesShort=['Jan.', 'Feb.', 'März', 'Apr.', 'Mai', 'Juni', 'Juli', 'Aug.', 'Sept.', 'Okt.', 'Nov.', 'Dez.'],
    numberDecimal=',',
    numberGroup='.',
    territory='DE',
    territoryName='Deutschland',
)


def test_locale():
    assert intl.locale('de').__dict__ == de
    assert intl.locale('de-DE').__dict__ == de
    assert intl.locale('de_DE').__dict__ == de


def test_date_format():
    d, t, n = intl.formatters(intl.locale('de_DE'))
    k = datetimex.new(2000, 1, 2, tz='utc')

    assert d.short(k) == '02.01.00'
    assert d.medium(k) == '02.01.2000'
    assert d.long(k) == '2. Januar 2000'


def test_time_format():
    d, t, n = intl.formatters(intl.locale('de_DE'))
    k = datetimex.new(2000, 1, 2, 3, 4, 5, tz='Europe/Berlin')

    assert t.short(k) == '03:04'
    assert t.medium(k) == '03:04:05'
    assert t.long(k) == '03:04:05 MEZ'

    k = datetimex.new(2000, 1, 2, 3, 4, 5, tz='Australia/Sydney')
    assert t.long(k) == '03:04:05 +1100'


def test_number_format():
    d, t, n = intl.formatters(intl.locale('de_DE'))

    assert n.decimal(12345678.90000) == '12345678,9'
    assert n.grouped(12345678.90000) == '12.345.678,9'
    assert n.currency(12345678.9000, currency='EUR') == '12.345.678,90\xA0€'
    assert n.percent(0.1234) == '12\xA0%'

    d, t, n = intl.formatters(intl.locale('en_US'))

    assert n.decimal(12345678.90000) == '12345678.9'
    assert n.grouped(12345678.90000) == '12,345,678.9'
    assert n.currency(12345678.9000, currency='$') == '$12,345,678.90'
