module.exports = v => ({
    '.searchSidebarIcon': {
        ...v.SIDEBAR_ICON('google:action/search')
    },

    '.searchClearButton.uiIconButton': {
        ...v.ICON_SIZE('small'),
        ...v.SVG(v.CLOSE_ICON, v.BORDER_COLOR),
    },

    '.searchWaitButton.uiIconButton': {
        ...v.ICON_SIZE('small'),
        // backgroundImage: v.IMAGE(__dirname + '/../../../ui/ajax.gif'),
    },

    '.searchBox': {
        '.uiInput .uiControlBox': {
            borderWidth: 0
        },
    },

    '.searchIcon.uiIconButton': {
        ...v.ICON_SIZE('normal'),
        ...v.SVG(v.SEARCH_ICON, v.BORDER_COLOR),
    },

    '.searchAltbar': {
        backgroundColor: v.COLOR.white,
        border: [1, 'solid', v.BORDER_COLOR],
    },

    '.searchSideButton': {
        minWidth: v.CONTROL_SIZE,
        maxWidth: v.CONTROL_SIZE,
    },

    '.searchAltbarResults .searchResults': {
        position: 'absolute',
        left: 0,
        right: 0,
        borderTopWidth: 0,
        maxHeight: 300,
        overflow: 'auto',
        overflowX: 'hidden',
        ...v.SHADOW,
    },

    '.searchResults .cmpListContent': {
        padding: v.UNIT4,
    },

    '.searchSidebar': {
        '.modSidebarTabHeader': {
            padding: [v.UNIT, v.UNIT2, v.UNIT, v.UNIT2],
        }
    },

    '.searchResultsFeatureText': {
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
