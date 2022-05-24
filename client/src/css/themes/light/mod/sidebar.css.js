module.exports = v => ({


    '.modSidebar': {
        position: 'absolute',
        top: 0,
        bottom: 0,
        width: '100%',
        backgroundColor: v.SIDEBAR_BODY_BACKGROUND,
        display: 'flex',
        flexDirection: 'column',
        zIndex: 3,
        ...v.SHADOW,
        ...v.TRANSITION('left', 'padding-bottom'),
    },

    '.modSidebarHeaderButton': {
        marginLeft: v.UNIT2,
        ...v.TRANSITION('all'),
        backgroundColor: v.SIDEBAR_HEADER_BACKGROUND,
        borderRadius: v.CONTROL_SIZE,
    },

    '.modSidebarHeaderButton.isActive': {
        opacity: 1,
        backgroundColor: v.SIDEBAR_ACTIVE_BUTTON_BACKGROUND,
    },

    '.modSidebarHeaderButton.isDisabled': {
        opacity: 0.4,
    },

    '.modSidebarCloseButton': {
        opacity: 0.85,
        ...v.SVG(v.CLOSE_ICON, v.SIDEBAR_HEADER_COLOR),
    },

    '.modSidebarOpenButton': {
        position: 'absolute',
        left: v.UNIT2,
        top: v.UNIT4,
        backgroundColor: v.SIDEBAR_OPEN_BUTTON_BACKGROUND,
        borderRadius: v.BORDER_RADIUS,
        ...v.SVG('google:navigation/menu', v.SIDEBAR_OPEN_BUTTON_COLOR),
    },


    '.modSidebar.isVisible': {
        left: 0,
    },

    '.modSidebarHeader': {
        backgroundColor: v.SIDEBAR_HEADER_BACKGROUND,
        padding: [
            v.UNIT4,
            v.UNIT4,
            v.UNIT4,
            v.UNIT2,
        ],
    },

    '.modSidebarTab': {
        flex: 1,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
    },

    '.modSidebarEmptyTab, .modSidebarEmptyTabBody': {
        textAlign: 'center',
        color: v.DISABLED_COLOR,
        padding: [v.UNIT2 * 8, v.UNIT4, v.UNIT4, v.UNIT4],
        lineHeight: 1.3,

        'a': {
            display: 'block',
            marginTop: 30,
            color: v.FOCUS_COLOR,
            cursor: 'pointer',
        }
    },

    '.modSidebarTabHeader': {
        borderBottom: [1, 'solid', v.BORDER_COLOR],
        padding: v.UNIT4,
        minHeight: v.CONTROL_SIZE,
        '.uiTitle': {
            fontSize: v.BIG_FONT_SIZE,
        },
    },

    '.modSidebarTabFooter': {
        minHeight: v.CONTROL_SIZE,

    },

    '.modSidebarAuxToolbar': {
        backgroundColor: v.SIDEBAR_AUX_TOOLBAR_BACKGROUND,
        paddingRight: v.UNIT2,
        paddingLeft: v.UNIT2,

        '.uiIconButton': {
            ...v.ICON_SIZE('small'),
        },
        '.modSidebarAuxCloseButton': {
            ...v.SIDEBAR_AUX_BUTTON(v.CLOSE_ICON),
        },
        '.cmpFeatureTaskButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/settings'),
        },


    },

    '.modSidebarTabBody, .modSidebarEmptyTabBody': {
        flex: 1,
        overflow: 'auto',
        padding: v.UNIT4,
    },

    '.modSidebarOverflowButton.uiIconButton': {
        ...v.SVG('google:navigation/more_horiz', v.TOOLBAR_BUTTON_COLOR),
        ...v.TRANSITION(),
        '&.isActive': {
            transform: 'rotate(90deg)',

        },
    },

    '.modSidebarOverflowPopup': {
        top: v.UNIT8 + v.CONTROL_SIZE,
        padding: [v.UNIT2, 0, v.UNIT2, 0],
        width: v.SIDEBAR_POPUP_WIDTH,
        backgroundColor: v.SIDEBAR_HEADER_BACKGROUND,
        color: v.SIDEBAR_HEADER_COLOR,
        cursor: 'default',
        userSelect: 'none',


        '.uiCell': {
            padding: v.UNIT2,
        },
    },


});
