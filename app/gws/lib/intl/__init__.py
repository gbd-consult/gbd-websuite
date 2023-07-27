"""Intl and localization tools."""

import datetime
import babel
import babel.dates
import babel.numbers
import pycountry

import gws
import gws.types as t


def locale(locale_uid: str) -> t.Optional[gws.Locale]:
    def f():
        if not locale_uid:
            return None

        locale_name = locale_uid.lower().strip().replace('-', '_')

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
        lo.numberDecimal = p.number_symbols['decimal']
        lo.numberGroup = p.number_symbols['group']

        return lo

    return gws.get_app_global(f'gws.lib.intl.locale.{locale_uid}', f)


def bibliographic_name(language: str) -> str:
    def f():
        if not language:
            return ''
        lang = pycountry.languages.get(alpha_2=language.lower())
        if not lang:
            return ''
        return lang.bibliographic

    return gws.get_app_global(f'gws.lib.intl.bibliographic_name.{language}', f)


class DateTimeFormat(t.Enum):
    short = 'short'
    medium = 'medium'
    long = 'long'
    iso = 'iso'


class DateFormatter:
    def __init__(self, locale_uid):
        self.localeUid = locale_uid

    def format(self, fmt: DateTimeFormat, d=None):
        if not d:
            d = datetime.datetime.now()
        elif isinstance(d, str):
            d = babel.dates.parse_date(d, self.localeUid)
        if fmt == DateTimeFormat.iso:
            return d.isoformat()
        return babel.dates.format_date(d, locale=self.localeUid, format=str(fmt))

    @property
    def short(self):
        return self.format(DateTimeFormat.short)

    @property
    def medium(self):
        return self.format(DateTimeFormat.medium)

    @property
    def long(self):
        return self.format(DateTimeFormat.long)

    @property
    def iso(self):
        return self.format(DateTimeFormat.iso)


class TimeFormatter:
    def __init__(self, locale_uid):
        self.localeUid = locale_uid

    def format(self, fmt: DateTimeFormat, d=None) -> str:
        d = babel.dates.parse_time(d, self.localeUid) if d else datetime.datetime.now()
        if fmt == DateTimeFormat.iso:
            return d.isoformat()
        return babel.dates.format_time(d, locale=self.localeUid, format=str(fmt))

    @property
    def short(self):
        return self.format(DateTimeFormat.short)

    @property
    def medium(self):
        return self.format(DateTimeFormat.medium)

    @property
    def long(self):
        return self.format(DateTimeFormat.long)

    @property
    def iso(self):
        return self.format(DateTimeFormat.iso)


class NumberFormat(t.Enum):
    decimal = 'decimal'
    grouped = 'grouped'
    currency = 'currency'
    percent = 'percent'


class NumberFormatter:
    def __init__(self, locale_uid):
        self.localeUid = locale_uid

    def format(self, fmt: NumberFormat, n, **kwargs) -> str:
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
    return gws.get_app_global(f'gws.lib.intl.date_formatter.{locale_uid}', lambda: DateFormatter(locale_uid))


def time_formatter(locale_uid) -> TimeFormatter:
    return gws.get_app_global(f'gws.lib.intl.time_formatter.{locale_uid}', lambda: TimeFormatter(locale_uid))


def number_formatter(locale_uid) -> NumberFormatter:
    return gws.get_app_global(f'gws.lib.intl.number_formatter.{locale_uid}', lambda: NumberFormatter(locale_uid))
