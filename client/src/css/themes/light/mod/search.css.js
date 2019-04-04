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
        '.uiInput .uiControlBox': {
            borderWidth: 0
        },
    },

    '.modSearchIcon.uiIconButton': {
        ...v.ICON_SIZE('normal'),
        ...v.SVG(v.SEARCH_ICON, v.BORDER_COLOR),
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
        ...v.SHADOW,
    },

    '.modSearchResults .cmpListContent': {
        padding: v.UNIT4,
    },

    '.modSearchSidebar': {
        '.modSidebarTabHeader': {
            padding: [v.UNIT, v.UNIT2, v.UNIT, v.UNIT2],
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
