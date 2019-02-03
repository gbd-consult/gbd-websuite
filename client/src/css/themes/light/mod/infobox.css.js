module.exports = v => ({
    '.modInfobox': {
        zIndex: 2,
        position: 'absolute',
        background: v.INFOBOX_BACKGROUND,
        display: 'flex',
        opacity: 0,

        ...v.TRANSITION('all'),
        ...v.SHADOW,
        maxHeight: '65%',
        minHeight: 90,

        '&.isActive': {
            opacity: 1,
        }

    },

    '.withSidebar .modInfobox': {},


    '.modInfoboxFooter': {
        padding: [0, v.UNIT4, 0, v.UNIT4],
        '.uiIconButton': {
            ...v.ICON('small'),
        }
    },

    '.modInfoboxCloseButton.uiIconButton': {
        ...v.SVG(v.CLOSE_ICON, v.INFOBOX_BUTTON_COLOR),
    },

    '.modInfoboxPagerBack.uiIconButton': {
        ...v.SVG('google:navigation/chevron_left', v.INFOBOX_BUTTON_COLOR),
    },

    '.modInfoboxPagerForward.uiIconButton': {
        ...v.SVG('google:navigation/chevron_right', v.INFOBOX_BUTTON_COLOR),
    },

    '.modInfobox .cmpFeatureTaskButton': {
        ...v.SVG('google:action/settings', v.INFOBOX_BUTTON_COLOR),

    },

    '.modInfoboxPagerText': {
        fontSize: v.SMALL_FONT_SIZE,
    },


    '.modInfoboxContent': {
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
    },

    '.modInfoboxBody': {
        flex: 1,
        overflow: 'auto',
        padding: v.UNIT8,
    },
});
