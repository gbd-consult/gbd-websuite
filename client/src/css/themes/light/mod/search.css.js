module.exports = v => ({
    '.modSearchSidebarIcon': {
        ...v.SVG('google:action/search', v.SIDEBAR_HEADER_COLOR)
    },

    '.modSearchClearButton.uiIconButton': {
        ...v.ICON('small'),
        ...v.SVG(v.CLOSE_ICON, v.BORDER_COLOR),
    },

    '.modSearchWaitButton.uiIconButton': {
        ...v.ICON('small'),
        backgroundImage: v.IMAGE('ajax.gif'),
    },

    '.modSearchBox': {
        '.uiInput .uiControlBox': {
            borderWidth: 0
        },
    },

    '.modSearchIcon': {
        ...v.ICON('normal'),
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
        ...v.SHADOW,
        overflow: 'auto',
        '.cmpFeatureListContent': {
            padding: v.UNIT4,
        }
    },

    '.modSearchSidebar': {
        '.modSidebarTabHeader': {
            padding: [v.UNIT, v.UNIT2, v.UNIT, v.UNIT2],
        }
    },

    '.modSearchResultsFeatureTitle': {
        fontWeight: 700,
        margin: [v.UNIT, 0, v.UNIT, 0],
    },

    '.modSearchResultsFeatureText': {
        fontSize: v.SMALL_FONT_SIZE,
        cursor: 'pointer',
    },

});
