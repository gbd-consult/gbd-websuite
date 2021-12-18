module.exports = v => ({
    '.modSearchSidebarIcon': {
        ...v.SIDEBAR_ICON('google:action/search')
    },

    '.modSearchClearButton.uiIconButton': {
        ...v.ICON_SIZE('small'),
        ...v.SVG(v.CLOSE_ICON, v.BORDER_COLOR),
    },

    '.modSearchWaitButton.uiIconButton': {
        ...v.ICON_SIZE('small'),
        backgroundImage: v.IMAGE('ajax.gif'),
    },

    '.modSearchBox': {
        '.uiControlBox': {
            borderWidth: 0
        },
    },

    '.modSearchOptions': {
        maxHeight: 0,
        overflow: 'hidden',
        ...v.TRANSITION(),

        '.isOpen&': {
            backgroundColor: v.EVEN_STRIPE_COLOR,
            maxHeight: 600,
        }

    },

    '.modSearchIcon.uiIconButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG(v.SEARCH_ICON, v.BORDER_COLOR),
        ...v.TRANSITION(),

        '.withOptions&': {
            ...v.SVG('google:navigation/chevron_right', v.BORDER_COLOR),
            transform: 'rotate(90deg)',
        },

        '.withOptions.isOpen&': {
            transform: 'rotate(-90deg)',
        }

    },

    '.modSearchAltbar': {
        backgroundColor: v.COLOR.white,
        border: [1, 'solid', v.BORDER_COLOR],
    },

    '.modSearchSideButton': {
        minWidth: v.CONTROL_SIZE,
        maxWidth: v.CONTROL_SIZE,
    },

    '.modSearchAltbarResults .modSearchResults': {
        position: 'absolute',
        left: 0,
        right: 0,
        borderTopWidth: 0,
        maxHeight: 300,
        overflow: 'auto',
        overflowX: 'hidden',
        border: [1, 'solid', v.BORDER_COLOR],
        borderTop: 'none',

        // ...v.SHADOW,
    },

    '.modSearchResults .cmpListContent': {
        padding: [v.UNIT4, v.UNIT4, v.UNIT4, 0],
    },

    '.modSearchSidebar': {
        '.modSidebarTabHeader': {
            padding: [v.UNIT, 0, 0, 0],
            '.modSearchBox .uiRow': {
                padding: [v.UNIT, v.UNIT2, v.UNIT, v.UNIT2],
            }
        },
        '.modSidebarTabBody': {
            padding: [0, 0, 0, 0],
        }
    },

    '.modSearchResultsFeatureText': {
        position: 'relative',
        'p.head': {
            color: v.FOCUS_COLOR,
            cursor: 'pointer',
            fontSize: v.NORMAL_FONT_SIZE,
            marginBottom: v.UNIT,
        },
        'p': {
            fontSize: v.SMALL_FONT_SIZE,
            lineHeight: '120%'
        },
    },

});
