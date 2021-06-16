module.exports = v => ({
    '.cmpInfobox': {
        zIndex: 2,
        position: 'absolute',
        background: v.INFOBOX_BACKGROUND,
        display: 'flex',
        opacity: 0,

        ...v.TRANSITION('all'),
        ...v.SHADOW,
        maxHeight: '65%',
        minHeight: 90,

        left: 0,
        right: 0,
        bottom: '-100%',

        [v.MEDIA('small+')]: {
            left: 'auto',
            right: v.UNIT8,
        },

        '&.isActive': {
            opacity: 1,
            bottom: v.INFOBAR_HEIGHT,
        }

    },

    '.withSidebar .cmpInfobox': {},

    '.cmpInfoboxFooter': {
        padding: [0, v.UNIT4, 0, v.UNIT4],
        '.uiIconButton': {
            ...v.ICON_SIZE('small'),
        }
    },

    '.cmpInfoboxCloseButton.uiIconButton': {
        ...v.SVG(v.CLOSE_ICON, v.INFOBOX_BUTTON_COLOR),
    },

    '.cmpInfoboxPagerBack.uiIconButton': {
        ...v.SVG('google:navigation/chevron_left', v.INFOBOX_BUTTON_COLOR),
    },

    '.cmpInfoboxPagerForward.uiIconButton': {
        ...v.SVG('google:navigation/chevron_right', v.INFOBOX_BUTTON_COLOR),
    },

    '.cmpInfobox .cmpFeatureTaskButton': {
        ...v.SVG('google:action/settings', v.INFOBOX_BUTTON_COLOR),

    },

    '.cmpInfoboxPagerText': {
        fontSize: v.SMALL_FONT_SIZE,
    },

    '.cmpInfoboxContent': {
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
    },

    '.cmpInfoboxBody': {
        flex: 1,
        overflow: 'auto',
        padding: v.UNIT8,

        '.cmpDescription .text': {
            maxWidth: 300,
        },
    },
});
