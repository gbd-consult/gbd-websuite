"""Intl and localization tools."""

import babel

import gws.types as t


class LocaleData(t.Data):
    id: str
    dateFormatLong: str
    dateFormatMedium: str
    dateFormatShort: str
    dateUnits: str  #: date unit names, e.g. 'YMD' for 'en', 'JMT' for 'de'
    daysLong: t.List[str]
    daysShort: t.List[str]
    firstWeekDay: int
    language: str
    languageName: str
    monthsLong: t.List[str]
    monthsShort: t.List[str]
    numberDecimal: str
    numberGroup: str


def locale_data(locale: str) -> t.Optional[LocaleData]:
    if not locale:
        return

    locale = locale.lower().strip().replace('-', '_')

    try:
        lo = babel.Locale.parse(locale, resolve_likely_subtags=True)
    except (ValueError, babel.UnknownLocaleError):
        return

    ld = LocaleData()

    # @TODO script etc
    ld.id = lo.language + '_' + lo.territory

    ld.dateFormatLong = str(lo.date_formats['long'])
    ld.dateFormatMedium = str(lo.date_formats['medium'])
    ld.dateFormatShort = str(lo.date_formats['short'])
    ld.daysLong = list(lo.days['format']['wide'].values())
    ld.daysShort = list(lo.days['format']['abbreviated'].values())
    ld.firstWeekDay = lo.first_week_day
    ld.language = lo.language
    ld.languageName = lo.language_name
    ld.monthsLong = list(lo.months['format']['wide'].values())
    ld.monthsShort = list(lo.months['format']['abbreviated'].values())
    ld.numberDecimal = lo.number_symbols['decimal']
    ld.numberGroup = lo.number_symbols['group']
    ld.dateUnits = (
            lo.unit_display_names['duration-year']['narrow'] +
            lo.unit_display_names['duration-month']['narrow'] +
            lo.unit_display_names['duration-day']['narrow'])

    return ld
