"""Intl and localization tools."""

import datetime
import babel
import babel.dates
import babel.numbers
import pycountry

import gws

_DEFAULT_UID = 'en_CA'  # English with metric units


# NB in the following code, `name` is a locale or language name (`de` or `de_DE`), `uid` is strictly a uid (`de_DE`)

def get_default_locale():
    """Returns the default locale object (``en_CA``)."""

    return get_locale(_DEFAULT_UID, fallback=False)


def get_locale(name: str, allowed: list[str] = None, fallback: bool = True) -> gws.Locale:
    """Locates a Locale object by locale name.

    If the name is invalid, and ``fallback`` is ``True``, return the first ``allowed`` locale,
    or the default locale. Otherwise, raise an exception.

    Args:
        name: Language or locale name like ``de`` or ``de_DE``.
        allowed: A list of allowed locale uids.
        fallback: Fall back to the default locale.
    """

    lo = _get_locale_by_name(name, allowed)
    if lo:
        return lo

    if not fallback:
        raise gws.Error(f'locale {name!r} not found')

    if allowed:
        lo = _get_locale_by_uid(allowed[0])
        if lo:
            return lo

    return get_default_locale()


def _get_locale_by_name(name, allowed):
    if not name:
        return

    name = name.strip().replace('-', '_')

    if '_' in name:
        # name is a uid
        if allowed and name not in allowed:
            return
        return _get_locale_by_uid(name)

    # just a lang name, try to find an allowed locale for this lang
    if allowed:
        for uid in allowed:
            if uid.startswith(name):
                return _get_locale_by_uid(uid)

    # try to get a generic locale
    return _get_locale_by_uid(name + '_zz')


def _get_locale_by_uid(uid):
    def _get():
        p = babel.Locale.parse(uid, resolve_likely_subtags=True)

        lo = gws.Locale()

        # @TODO script etc
        lo.uid = p.language + '_' + p.territory

        lo.language = p.language
        lo.languageName = p.language_name

        lg = pycountry.languages.get(alpha_2=lo.language)
        if not lg:
            raise ValueError(f'unknown language {lo.language}')
        lo.language3 = getattr(lg, 'alpha_3', '')
        lo.languageBib = getattr(lg, 'bibliographic', lo.language3)
        lo.languageNameEn = getattr(lg, 'name', lo.languageName)

        lo.territory = p.territory
        lo.territoryName = p.territory_name

        lo.dateFormatLong = str(p.date_formats['long'])
        lo.dateFormatMedium = str(p.date_formats['medium'])
        lo.dateFormatShort = str(p.date_formats['short'])
        lo.dateUnits = (
                p.unit_display_names['duration-year']['narrow'] +
                p.unit_display_names['duration-month']['narrow'] +
                p.unit_display_names['duration-day']['narrow'])

        lo.dayNamesLong = list(p.days['format']['wide'].values())
        lo.dayNamesNarrow = list(p.days['format']['narrow'].values())
        lo.dayNamesShort = list(p.days['format']['abbreviated'].values())

        lo.firstWeekDay = p.first_week_day

        lo.monthNamesLong = list(p.months['format']['wide'].values())
        lo.monthNamesNarrow = list(p.months['format']['narrow'].values())
        lo.monthNamesShort = list(p.months['format']['abbreviated'].values())

        lo.numberDecimal = p.number_symbols['latn']['decimal']
        lo.numberGroup = p.number_symbols['latn']['group']

        return lo

    try:
        return gws.u.get_app_global(f'gws.lib.intl.locale.{uid}', _get)
    except (AttributeError, ValueError, babel.UnknownLocaleError):
        gws.log.exception()
        return None


##


class _FnStr:
    """Allow a property to act both as a method and as a string."""

    def __init__(self, method, arg):
        self.method = method
        self.arg = arg

    def __str__(self):
        return self.method(self.arg)

    def __call__(self, a=None):
        return self.method(self.arg, a)


# @TODO support RFC 2822

class DateFormatter(gws.DateFormatter):
    def __init__(self, locale: gws.Locale):
        self.locale = locale
        self.short = _FnStr(self.format, gws.DateTimeFormat.short)
        self.medium = _FnStr(self.format, gws.DateTimeFormat.medium)
        self.long = _FnStr(self.format, gws.DateTimeFormat.long)
        self.iso = _FnStr(self.format, gws.DateTimeFormat.iso)

    def format(self, fmt: gws.DateTimeFormat, date=None):
        if not date:
            date = datetime.datetime.now().date()
        elif isinstance(date, str):
            date = babel.dates.parse_date(date, self.locale.uid)
        else:
            raise gws.Error(f'invalid {date=}')
        if fmt == gws.DateTimeFormat.iso:
            return date.isoformat()
        return babel.dates.format_date(date, locale=self.locale.uid, format=str(fmt))


class TimeFormatter(gws.TimeFormatter):
    def __init__(self, locale: gws.Locale):
        self.locale = locale
        self.short = _FnStr(self.format, gws.DateTimeFormat.short)
        self.medium = _FnStr(self.format, gws.DateTimeFormat.medium)
        self.long = _FnStr(self.format, gws.DateTimeFormat.long)
        self.iso = _FnStr(self.format, gws.DateTimeFormat.iso)

    def format(self, fmt: gws.DateTimeFormat, time=None) -> str:
        if not time:
            time = datetime.datetime.now().time()
        elif isinstance(time, str):
            time = babel.dates.parse_time(time, self.locale.uid)
        else:
            raise gws.Error(f'invalid {time=}')
        if fmt == gws.DateTimeFormat.iso:
            return time.isoformat()
        return babel.dates.format_time(time, locale=self.locale.uid, format=str(fmt))


# @TODO scientific, compact...

class NumberFormatter(gws.NumberFormatter):
    def __init__(self, locale: gws.Locale):
        self.locale = locale
        self.fns = {
            gws.NumberFormat.decimal: self.decimal,
            gws.NumberFormat.grouped: self.grouped,
            gws.NumberFormat.currency: self.currency,
            gws.NumberFormat.percent: self.percent,
        }

    def format(self, fmt, n, *args, **kwargs):
        fn = self.fns.get(fmt)
        if not fn:
            return str(n)
        return fn(n, *args, **kwargs)

    def decimal(self, n, *args, **kwargs):
        return babel.numbers.format_decimal(n, locale=self.locale.uid, group_separator=False, *args, **kwargs)

    def grouped(self, n, *args, **kwargs):
        return babel.numbers.format_decimal(n, locale=self.locale.uid, group_separator=True, *args, **kwargs)

    def currency(self, n, currency, *args, **kwargs):
        return babel.numbers.format_currency(n, currency=currency, locale=self.locale.uid, *args, **kwargs)

    def percent(self, n, *args, **kwargs):
        return babel.numbers.format_percent(n, locale=self.locale.uid, *args, **kwargs)


##


def get_formatters(locale: gws.Locale) -> tuple[DateFormatter, TimeFormatter, NumberFormatter]:
    """Return a tuple of locale-aware formatters."""

    def _get():
        return (
            DateFormatter(locale),
            TimeFormatter(locale),
            NumberFormatter(locale),
        )

    return gws.u.get_app_global(f'gws.lib.intl.formatters.{locale.uid}', _get)
