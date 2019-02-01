module.exports = v => ({
    '.modPopup': {
        zIndex: 2,
        position: 'absolute',
        background: v.POPUP_BACKGROUND,
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

    '.withSidebar .modPopup': {
    },

    '.modPopupCloseButton': {
        '&.uiIconButton': {
            ...v.ICON('medium'),
            ...v.CLOSE_ICON(v.POPUP_BUTTON_COLOR),
        },
    },

    '.modPopupZoomButton': {
        '&.uiIconButton': {
            ...v.ICON('medium'),
            ...v.GOOGLE_SVG('image/center_focus_weak', v.POPUP_BUTTON_COLOR),
        },
    },

    '.modPopupLensButton': {
        '&.uiIconButton': {
            ...v.ICON('medium'),
            ...v.LOCAL_SVG('search_lens', v.POPUP_BUTTON_COLOR),
        },
    },

    '.modPopupSelectButton': {
        '&.uiIconButton': {
            ...v.ICON('medium'),
            ...v.LOCAL_SVG('select', v.POPUP_BUTTON_COLOR),
        },
    },

    '.modPopupPagerBack': {
        '&.uiIconButton': {
            ...v.ICON('medium'),
            ...v.GOOGLE_SVG('navigation/chevron_left', v.POPUP_BUTTON_COLOR),
        },
    },

    '.modPopupPagerForward': {
        '&.uiIconButton': {
            ...v.ICON('medium'),
            ...v.GOOGLE_SVG('navigation/chevron_right', v.POPUP_BUTTON_COLOR),
        },
    },

    '.modPopupPagerText': {
        fontSize: v.SMALL_FONT_SIZE,
    },


    '.modPopupContent': {
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
    },

    '.modPopupBody': {
        flex: 1,
        overflow: 'auto',
        padding: v.UNIT8,
    },

    '.modPopupFooter': {
        padding: [0, v.UNIT4, 0, v.UNIT8],
        // borderTopWidth: 1,
        // borderTopStyle: 'solid',
        // borderTopColor: v.BORDER_COLOR,


    }
});
