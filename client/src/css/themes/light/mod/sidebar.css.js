

module.exports = v => ({


    '.modSidebar': {
        position: 'absolute',
        top: 0,
        bottom: 0,
        width: '100%',
        backgroundColor: v.SIDEBAR_BODY_BACKGROUND,
        display: 'flex',
        flexDirection: 'column',
        ...v.SHADOW,
        ...v.TRANSITION('left'),
    },

    '.modSidebarHeaderButton': {
        marginLeft: v.UNIT2,
        ...v.TRANSITION('all')
    },

    '.modSidebarHeaderButton.isActive': {
        opacity: 1,
        backgroundColor: v.SIDEBAR_ACTIVE_BUTTON_BACKGROUND,
        borderRadius: v.CONTROL_SIZE,
    },

    '.modSidebarHeaderButton.isDisabled': {
        opacity: 0.4,
    },

    '.modSidebarCloseButton': {
        opacity: 0.85,
        ...v.CLOSE_SVG(v.SIDEBAR_HEADER_COLOR),
    },

    '.modSidebarOpenButton': {
        position: 'absolute',
        left: v.UNIT2,
        top: v.UNIT4,
        backgroundColor: v.SIDEBAR_OPEN_BUTTON_BACKGROUND,
        borderRadius: v.BORDER_RADIUS,
        ...v.GOOGLE_SVG('navigation/menu', v.SIDEBAR_OPEN_BUTTON_COLOR),
    },


    '.modSidebar.isVisible': {
        left: 0,
    },

    '.modSidebarHeader': {
        background: v.SIDEBAR_HEADER_BACKGROUND,
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

    '.modSidebarEmptyTab': {
        textAlign: 'center',
        color: v.DISABLED_COLOR,
        padding: [v.UNIT2 * 8, v.UNIT4, v.UNIT4, v.UNIT4],
    },

    '.modSidebarTabHeader': {
        borderBottomWidth: 1,
        borderBottomStyle: 'solid',
        borderBottomColor: v.BORDER_COLOR,
        padding: v.UNIT4,
        minHeight: v.CONTROL_SIZE,
        '.uiTitle': {
            fontSize: v.BIG_FONT_SIZE,
        },
    },

    '.modSidebarTabFooter': {
        borderTopWidth: 1,
        borderTopStyle: 'solid',
        borderTopColor: v.BORDER_COLOR,
        minHeight: v.CONTROL_SIZE,

    },

    '.modSidebarSecondaryToolbar': {
        '.uiIconButton': {
        ...v.ICON('medium'),
    },
        backgroundColor: v.SECONDARY_TOOLBAR_BACKGROUND,
        paddingRight: v.UNIT2,
    },


    '.modSidebarTabBody': {
        flex: 1,
        overflow: 'auto',
        padding: v.UNIT4,
    },



});
