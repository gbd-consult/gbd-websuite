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
        borderRadius: v.BORDER_RADIUS,
        backgroundColor: v.COLOR.white,
        border: [1, 'solid', v.BORDER_COLOR],
    },

    '.modSearchSideButton': {
        minWidth: v.CONTROL_SIZE,
        maxWidth: v.CONTROL_SIZE,
    },

    '.modSearchAltbarResults .modSearchResults': {
        position: 'absolute',
        left: v.UNIT4,
        right: v.UNIT4,
        borderTopWidth: 0,
        maxHeight: 300,
        overflow: 'auto',
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
        '.head': {
            color: v.FOCUS_COLOR,
            cursor: 'pointer',
        },
    },

});
