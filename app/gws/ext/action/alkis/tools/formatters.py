import gws
import gws.gis.feature
import gws.gis.shape
import gws.tools.sheet


def flurstueck_property_sheet(fs):
    s = gws.tools.sheet.Sheet(fs)

    s.section('Basisdaten')

    s.entry('Flurnummer', s.str('flurnummer'))
    s.entry('Zähler', s.str('zaehler'))
    s.entry('Nenner', s.str('nenner', default='-'))
    s.entry('Fläche', s.area('area.total'))
    s.entry('ges. Gebäudefläche', s.area('area.gebaeude'))
    s.entry('Flurstücksfläche abz. Gebäudefläche', s.area('area.diff'))

    s.section('Lage')

    s.entry('Gemeinde', s.str('gemeinde'))
    s.entry('Gemarkung', s.str('gemarkung'))

    for lage in s.list('lage'):
        s.entry('Adresse', s.format('{strasse} {hausnummer}', src=lage))

    s.section('Gebäudenachweis')

    for geb in s.list('gebaeude'):
        s.entry('Funktion', s.str('gebaeudefunktion', src=geb))
        s.entry('Fläche', s.area('area', src=geb))

    for bs in s.list('buchung'):

        s.section('Buchungssatz')

        s.entry('Buchungsart', s.str('buchungsart', src=bs))
        s.entry('Anteil', s.str('anteil', src=bs))
        s.entry('Buchungsblattnummer', s.str('buchungsblatt.buchungsblattnummermitbuchstabenerweiterung', src=bs))

        for eigentuemer in s.list('eigentuemer', src=bs):
            e = s.format('''
                {vorname} {nachnameoderfirma}
                {anschrift.strasse} {anschrift.hausnummer}
                {anschrift.postleitzahlpostzustellung} {anschrift.ort_post}
            ''', src=eigentuemer.get('person'))

            s.entry('Eigentümer\n' + s.str('anteil', src=eigentuemer), e)

    s.section('Nutzung')

    for nu in s.list('nutzung'):
        h = s.str('type', src=nu)
        if s.str('key', src=nu) != s.str('type', src=nu):
            h += ' (' + s.str('key', src=nu) + ')'
        s.entry(h, s.area('a_area', src=nu))

    return s
