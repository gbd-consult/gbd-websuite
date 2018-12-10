module.exports = v => ({
    '.modSearchSidebarIcon': {
        ...v.GOOGLE_SVG('action/search', v.SIDEBAR_HEADER_COLOR)
    },

    '.modSearchClearButton.uiIconButton': {
        ...v.ICON('small'),
        ...v.CLOSE_SVG(v.BORDER_COLOR),
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
        ...v.GOOGLE_SVG('action/search', v.BORDER_COLOR),
    },

    '.modSearchToolbar': {
        width: v.TOOLBAR_SEARCH_WIDTH + v.UNIT8,
        borderRadius: v.BORDER_RADIUS,
        backgroundColor: v.TOOLBAR_SEARCH_BACKGROUND,
        margin: [0, v.UNIT4, 0, v.UNIT4],
        //...v.SHADOW,
        border: '1px solid ' + v.BORDER_COLOR,
        zIndex: 1,

    },

    '.modSearchSideButton': {
        minWidth: v.CONTROL_SIZE,
        maxidth: v.CONTROL_SIZE,
    },

    '.modSearchToolbarResults .modSearchResults': {
        position: 'absolute',
        top: v.TOOLBAR_HEIGHT - v.UNIT2,
        right: v.UNIT8,
        borderTopWidth: 0,
        width: v.TOOLBAR_SEARCH_WIDTH,
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
