"""Intl and localization tools."""

import datetime
import babel
import babel.dates
import babel.numbers
import pycountry

import gws
import gws.types as t


def locale(locale_uid: str) -> t.Optional[gws.Locale]:
    """Creates a locale object with formatting information about date, time and numbers.

    Args:
        locale_uid: ID in the format `language_territory` e.g. `de_DE`, or `de`.

    Returns:
        Formatting information for that area.
    """

    def f():
        if not locale_uid:
            return None

        locale_name = locale_uid.lower().strip().replace('-', '_')

        if '_' not in locale_name:
            locale_name = locale_name + '_zz'

        try:
            p = babel.Locale.parse(locale_name, resolve_likely_subtags=True)
        except (ValueError, babel.UnknownLocaleError):
            return None

        lo = gws.Locale()

        # @TODO script etc
        lo.id = p.language + '_' + p.territory

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
        lo.language = p.language
        lo.languageName = p.language_name
        lo.monthNamesLong = list(p.months['format']['wide'].values())
        lo.monthNamesNarrow = list(p.months['format']['narrow'].values())
        lo.monthNamesShort = list(p.months['format']['abbreviated'].values())
        lo.numberDecimal = p.number_symbols['latn']['decimal']
        lo.numberGroup = p.number_symbols['latn']['group']

        return lo

    return gws.get_app_global(f'gws.lib.intl.locale.{locale_uid}', f)


def bibliographic_name(language: str) -> str:
    """Country abbreviation for a given language

    Args:
        language: 2 letter abbreviation for a language.

    Returns:
        ISO 3166-1 alpha-3 code for the language's country.
    """

    def f():
        if not language:
            return ''
        lang = pycountry.languages.get(alpha_2=language.lower())
        if not lang:
            return ''
        return lang.alpha_3

    return gws.get_app_global(f'gws.lib.intl.bibliographic_name.{language}', f)


class DateTimeFormat(t.Enum):
    """Enumeration indicating the length of the date/time format."""
    short = 'short'
    """Local short format."""
    medium = 'medium'
    """Local medium format."""
    long = 'long'
    """Local long format."""
    iso = 'iso'
    """ISO 8601 format."""


class DateFormatter:
    """Used for date formatting"""

    def __init__(self, locale_uid):
        self.localeUid = locale_uid

    def format(self, fmt: DateTimeFormat, d=None):
        """Formats the date with respect to the `locale_uid`.

        Args:
            fmt: Describes the length of the return.
            d: Date, if none is given the current date will be used as default.

        Returns:
            The date in the locale format.
        """
        if not d:
            d = datetime.datetime.now()
        elif isinstance(d, str):
            d = babel.dates.parse_date(d, self.localeUid)
        if fmt == DateTimeFormat.iso:
            return d.isoformat()
        return babel.dates.format_date(d, locale=self.localeUid, format=str(fmt))

    @property
    def short(self):
        """Returns the date in a short format with respect to the locale date format"""
        return self.format(DateTimeFormat.short)

    @property
    def medium(self):
        """Returns the date in a medium format with respect to the locale date format"""
        return self.format(DateTimeFormat.medium)

    @property
    def long(self):
        """Returns the date in a long format with respect to the locale date format"""
        return self.format(DateTimeFormat.long)

    @property
    def iso(self):
        """Returns the time and date in the ISO 8601 format."""
        return self.format(DateTimeFormat.iso)


class TimeFormatter:
    """Used for time formatting"""

    def __init__(self, locale_uid):
        self.localeUid = locale_uid

    def format(self, fmt: DateTimeFormat, d=None) -> str:
        """Formats the time with respect to the `locale_uid`.

        Args:
            fmt: Describes the length of the return.
            d: Time, if none is given the current time will be used as default.

        Returns:
            The time in the locale format.
        """
        d = babel.dates.parse_time(d, self.localeUid) if d else datetime.datetime.now()
        if fmt == DateTimeFormat.iso:
            return d.isoformat()
        return babel.dates.format_time(d, locale=self.localeUid, format=str(fmt))

    @property
    def short(self):
        """Returns the time in a short format with respect to the locale time format"""
        return self.format(DateTimeFormat.short)

    @property
    def medium(self):
        """Returns the time in a medium format with respect to the locale time format"""
        return self.format(DateTimeFormat.medium)

    @property
    def long(self):
        """Returns the time in a long format with respect to the locale time format"""
        return self.format(DateTimeFormat.long)

    @property
    def iso(self):
        """Returns the time and date in the ISO 8601 format."""
        return self.format(DateTimeFormat.iso)


class NumberFormat(t.Enum):
    """Enumeration indicating the number format."""
    decimal = 'decimal'
    """Locale decimal format."""
    grouped = 'grouped'
    """Locale grouped format."""
    currency = 'currency'
    """Locale currency format"""
    percent = 'percent'
    """Locale percent format."""


class NumberFormatter:
    """Used for number formatting"""

    def __init__(self, locale_uid):
        self.localeUid = locale_uid

    def format(self, fmt: NumberFormat, n, **kwargs) -> str:
        """Formats the number with respect to the `locale_uid`.

        Args:
            fmt: Describes the format of the return.
            n: Number.
            kwargs: Passes the currency parameter forward.

        Returns:
            The number in the locale format.
        """
        if fmt == NumberFormat.decimal:
            return babel.numbers.format_decimal(n, locale=self.localeUid, group_separator=False, **kwargs)
        if fmt == NumberFormat.grouped:
            return babel.numbers.format_decimal(n, locale=self.localeUid, group_separator=True, **kwargs)
        if fmt == NumberFormat.currency:
            return babel.numbers.format_currency(n, locale=self.localeUid, **kwargs)
        if fmt == NumberFormat.percent:
            return babel.numbers.format_percent(n, locale=self.localeUid, **kwargs)
        return str(n)


def date_formatter(locale_uid) -> DateFormatter:
    """Creates a `DateFormatter` if there is no other instance of that class to the same `locale_uid`.

    Args:
        locale_uid: ID in the format `language_territory` e.g. `de_DE`.

    Returns:
        `DateFormatter` object.
    """
    return gws.get_app_global(f'gws.lib.intl.date_formatter.{locale_uid}', lambda: DateFormatter(locale_uid))


def time_formatter(locale_uid) -> TimeFormatter:
    """Creates a `TimeFormatter` if there is no other instance of that class to the same `locale_uid`.

        Args:
            locale_uid: ID in the format `language_territory` e.g. `de_DE`.

        Returns:
            `TimeFormatter` object.
        """
    return gws.get_app_global(f'gws.lib.intl.time_formatter.{locale_uid}', lambda: TimeFormatter(locale_uid))


def number_formatter(locale_uid) -> NumberFormatter:
    """Creates a `NumberFormatter` if there is no other instance of that class to the same `locale_uid`.

        Args:
            locale_uid: ID in the format `language_territory` e.g. `de_DE`.

        Returns:
            `NumberFormatter` object.
        """
    return gws.get_app_global(f'gws.lib.intl.number_formatter.{locale_uid}', lambda: NumberFormatter(locale_uid))
