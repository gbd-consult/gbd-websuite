"""Locales and locale-aware formatting.

Builds ``gws.Locale`` objects from ``babel`` and ``pycountry`` data and provides
locale-aware formatters for dates, times and numbers.

``locale`` finds a locale by a language or locale name (``de`` or ``de_DE``), optionally
restricted to a list of allowed locale uids, and falls back to the first allowed locale
or to the default locale ``en_CA`` (English with metric units). ``formatters`` returns a
``DateFormatter``, ``TimeFormatter`` and ``NumberFormatter`` for a locale. Locales and
formatters are cached per application.

Example::

    loc = gws.lib.intl.locale('de', allowed=['de_DE', 'en_US'])
    date_fmt, time_fmt, num_fmt = gws.lib.intl.formatters(loc)
    date_fmt.long('2013-12-11')        # '11. Dezember 2013'
    num_fmt.grouped(1234567.5)         # '1.234.567,5'
"""

import babel
import babel.dates
import babel.numbers
import pycountry

import gws
import gws.lib.datetimex

_DEFAULT_UID = 'en_CA'  # English with metric units


# NB in the following code, `name` is a locale or language name (`de` or `de_DE`), `uid` is strictly a uid (`de_DE`)


def default_locale():
    """Return the default locale ``en_CA``.

    Returns:
        A Locale object.
    """

    return locale(_DEFAULT_UID, fallback=False)


def locale(name: str | None, allowed: list[str] = None, fallback: bool = True) -> gws.Locale:
    """Find a locale by a language or locale name.

    A language name selects the first allowed locale for that language or, if there is none,
    the most likely locale for the language (``de`` becomes ``de_DE``). A locale name must be allowed. If no locale is found and ``fallback`` is ``True``,
    the first ``allowed`` locale or the default locale is returned.

    Args:
        name: Language or locale name like ``de``, ``de_DE`` or ``de-DE``.
        allowed: Allowed locale uids.
        fallback: If ``True``, fall back to a default instead of raising an error.

    Returns:
        A Locale object.

    Raises:
        ``gws.Error``: If no locale is found and ``fallback`` is ``False``.
    """

    lo = _locale_by_name(name, allowed)
    if lo:
        return lo

    if not fallback:
        raise gws.Error(f'locale {name!r} not found')

    if allowed:
        lo = _locale_by_uid(allowed[0])
        if lo:
            return lo

    return default_locale()


def _locale_by_name(name, allowed):
    """Find a locale by a language or locale name, or return ``None``."""
    if not name:
        return

    name = name.strip().replace('-', '_')

    if '_' in name:
        # name is a uid
        if allowed and name not in allowed:
            return
        return _locale_by_uid(name)

    # just a lang name, try to find an allowed locale for this lang
    if allowed:
        for uid in allowed:
            if uid.startswith(name):
                return _locale_by_uid(uid)

    # try to get a generic locale
    return _locale_by_uid(name + '_zz')


def _locale_by_uid(uid):
    """Create a cached locale for a uid, or return ``None`` if the uid is unknown."""
    def _get():
        p = babel.Locale.parse(uid, resolve_likely_subtags=True)

        lo = gws.Locale()

        # @TODO script etc
        terr = p.territory or 'ZZ'
        lo.uid = p.language + '_' + terr

        lo.language = p.language
        lo.languageName = p.language_name

        lg = pycountry.languages.get(alpha_2=lo.language)
        if not lg:
            raise ValueError(f'unknown language {lo.language}')
        lo.language3 = getattr(lg, 'alpha_3', '')
        lo.languageBib = getattr(lg, 'bibliographic', lo.language3)
        lo.languageNameEn = getattr(lg, 'name', lo.languageName)

        lo.territory = terr
        lo.territoryName = p.territory_name

        lo.dateFormatLong = str(p.date_formats['long'])
        lo.dateFormatMedium = str(p.date_formats['medium'])
        lo.dateFormatShort = str(p.date_formats['short'])
        lo.dateUnits = (
            p.unit_display_names['duration-year']['narrow']
            + p.unit_display_names['duration-month']['narrow']
            + p.unit_display_names['duration-day']['narrow']
        )

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
    """Date formatter, based on ``babel.dates``."""

    def __init__(self, loc: gws.Locale):
        """Create a date formatter.

        ``short``, ``medium``, ``long`` and ``iso`` can be called with a date, or used as strings,
        which format the current date.

        Args:
            loc: Locale to format for.
        """
        self.locale = loc
        self.short = _FnStr(self.format, gws.DateTimeFormat.short)
        self.medium = _FnStr(self.format, gws.DateTimeFormat.medium)
        self.long = _FnStr(self.format, gws.DateTimeFormat.long)
        self.iso = _FnStr(self.format, gws.DateTimeFormat.iso)

    def format(self, fmt: gws.DateTimeFormat, date=None):
        date = date or gws.lib.datetimex.now()
        d = gws.lib.datetimex.parse(date)
        if not d:
            raise gws.Error(f'invalid {date=}')
        if fmt == gws.DateTimeFormat.iso:
            return gws.lib.datetimex.to_iso_date_string(d)
        return babel.dates.format_date(d, locale=self.locale.uid, format=str(fmt))


class TimeFormatter(gws.TimeFormatter):
    """Time formatter, based on ``babel.dates``."""

    def __init__(self, loc: gws.Locale):
        """Create a time formatter.

        ``short``, ``medium``, ``long`` and ``iso`` can be called with a time, or used as strings,
        which format the current time.

        Args:
            loc: Locale to format for.
        """
        self.locale = loc
        self.short = _FnStr(self.format, gws.DateTimeFormat.short)
        self.medium = _FnStr(self.format, gws.DateTimeFormat.medium)
        self.long = _FnStr(self.format, gws.DateTimeFormat.long)
        self.iso = _FnStr(self.format, gws.DateTimeFormat.iso)

    def format(self, fmt: gws.DateTimeFormat, date=None) -> str:
        date = date or gws.lib.datetimex.now()
        d = gws.lib.datetimex.parse(date) or gws.lib.datetimex.parse_time(date)
        if not d:
            raise gws.Error(f'invalid {date=}')
        if fmt == gws.DateTimeFormat.iso:
            return gws.lib.datetimex.to_iso_time_string(d)
        return babel.dates.format_time(d, locale=self.locale.uid, format=str(fmt))


# @TODO scientific, compact...


class NumberFormatter(gws.NumberFormatter):
    """Number formatter, based on ``babel.numbers``."""

    def __init__(self, loc: gws.Locale):
        """Create a number formatter.

        Args:
            loc: Locale to format for.
        """
        self.locale = loc
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


def formatters(loc: gws.Locale) -> tuple[DateFormatter, TimeFormatter, NumberFormatter]:
    """Return the formatters for a locale.

    Formatters are cached per locale.

    Args:
        loc: A Locale object.

    Returns:
        A date, a time and a number formatter.
    """

    def _get():
        return (
            DateFormatter(loc),
            TimeFormatter(loc),
            NumberFormatter(loc),
        )

    return gws.u.get_app_global(f'gws.lib.intl.formatters.{loc.uid}', _get)
