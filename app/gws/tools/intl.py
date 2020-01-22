"""Intl and localization tools."""

import babel

import gws.types as t


class LocaleData(t.Data):
    id: str
    daysAbbreviated: t.List[str]
    daysWide: t.List[str]
    firstWeekDay: int
    language: str
    languageName: str
    monthsAbbreviated: t.List[str]
    monthsWide: t.List[str]
    numberDecimal: str
    numberGroup: str


def locale_data(locale: str) -> t.Optional[LocaleData]:
    locale = locale.lower().strip().replace('-', '_')

    try:
        lo = babel.Locale.parse(locale, resolve_likely_subtags=True)
    except (ValueError, babel.UnknownLocaleError):
        return

    ld = LocaleData()

    # @TODO script etc
    ld.id = lo.language + '_' + lo.territory

    ld.daysAbbreviated = list(lo.days['format']['abbreviated'].values())
    ld.daysWide = list(lo.days['format']['wide'].values())
    ld.firstWeekDay = lo.first_week_day
    ld.language = lo.language
    ld.languageName = lo.language_name
    ld.monthsAbbreviated = list(lo.months['format']['abbreviated'].values())
    ld.monthsWide = list(lo.months['format']['wide'].values())
    ld.numberDecimal = lo.number_symbols['decimal']
    ld.numberGroup = lo.number_symbols['group']

    return ld
