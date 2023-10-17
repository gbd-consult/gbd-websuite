"""Global vars for the in-container test runner."""

import gws

DEFAULT_MANIFEST = {
    'withStrictConfig': True,
    'withFallbackConfig': False,
}

SESSION_STORE_PATH = gws.VAR_DIR + '/test_session_store.sqlite'

GWS_CONFIG_PATH = gws.VAR_DIR + '/test_gws_config.json'

# global config

CONFIG = {}


# some points for mock features

class POINTS:
    # PT Passy
    paris = [254451, 6250716]

    # PT Maxplatz
    dus = [753834, 6660874]

    # Linden x Birken Str
    dus1 = [756871, 6661810]

    # PT Wehrhahn
    dus2 = [756766, 6661801]

    # Linden x Mendelssohn Str
    dus3 = [757149, 6661832]

    # PT Neßler Str.
    dus4 = [765513, 6648529]

    # PT Gärdet
    stockholm = [2014778, 8255502]

    # PT Ustinksy Most
    moscow = [4189555, 7508535]

    # PT Cho Ba Chieu / Gia Dinh
    vietnam = [11877461, 1209716]

    # PT Flemington Racecourse / Melbourne
    australia = [16131032, -4549421]

    # Yarawa Rd x Namara Rd
    fiji = [19865901, -2052085]

    # Main Road Y junction
    pitcairn = [-14482452, -2884039]

    # PT Allende
    mexico = [-11035867, 2206279]

    # Park Av x Carson St
    memphis = [-10014603, 4178550]

    # PT Broadway & West 3rd
    ny = [-8237102, 4972223]

    # PT Lime Str
    liverpool = [-331463, 7058753]

    # PT East India Dock Rd
    london = [-48, 6712663]

    # PT Tema Harbour
    ghana = [201, 627883]
