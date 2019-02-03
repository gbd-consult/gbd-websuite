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

    '.withSidebar .modInfobox': {
    },

    '.modInfoboxCloseButton': {
        '&.uiIconButton': {
            ...v.ICON('medium'),
            ...v.SVG(v.CLOSE_ICON, v.INFOBOX_BUTTON_COLOR),
        },
    },

    '.modInfoboxZoomButton': {
        '&.uiIconButton': {
            ...v.ICON('medium'),
            ...v.SVG('google:image/center_focus_weak', v.INFOBOX_BUTTON_COLOR),
        },
    },

    '.modInfoboxLensButton': {
        '&.uiIconButton': {
            ...v.ICON('medium'),
            ...v.SVG('search_lens', v.INFOBOX_BUTTON_COLOR),
        },
    },

    '.modInfoboxSelectButton': {
        '&.uiIconButton': {
            ...v.ICON('medium'),
            ...v.SVG('select', v.INFOBOX_BUTTON_COLOR),
        },
    },

    '.modInfoboxPagerBack': {
        '&.uiIconButton': {
            ...v.ICON('medium'),
            ...v.SVG('google:navigation/chevron_left', v.INFOBOX_BUTTON_COLOR),
        },
    },

    '.modInfoboxPagerForward': {
        '&.uiIconButton': {
            ...v.ICON('medium'),
            ...v.SVG('google:navigation/chevron_right', v.INFOBOX_BUTTON_COLOR),
        },
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

    '.modInfoboxFooter': {
        padding: [0, v.UNIT4, 0, v.UNIT8],
        // borderTopWidth: 1,
        // borderTopStyle: 'solid',
        // borderTopColor: v.BORDER_COLOR,


    }
});
