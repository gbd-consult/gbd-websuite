"""Intl and localization tools."""

import babel

import gws
import gws.types as t


class Locale(gws.Data):
    id: str
    dateFormatLong: str
    dateFormatMedium: str
    dateFormatShort: str
    dateUnits: str  #: date unit names, e.g. 'YMD' for 'en', 'JMT' for 'de'
    dayNamesLong: t.List[str]
    dayNamesShort: t.List[str]
    dayNamesNarrow: t.List[str]
    firstWeekDay: int
    language: str
    languageName: str
    monthNamesLong: t.List[str]
    monthNamesShort: t.List[str]
    monthNamesNarrow: t.List[str]
    numberDecimal: str
    numberGroup: str


def locale(locale_uid: str) -> t.Optional[Locale]:
    if not locale_uid:
        return None

    locale_name = locale_uid.lower().strip().replace('-', '_')

    try:
        p = babel.Locale.parse(locale_name, resolve_likely_subtags=True)
    except (ValueError, babel.UnknownLocaleError):
        return None

    lo = Locale()

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
