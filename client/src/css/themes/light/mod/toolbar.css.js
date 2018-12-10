

module.exports = v => ({

    '.modToolbar': {
        alignItems: 'center',
        backgroundColor: v.TOOLBAR_BACKGROUND,
        borderRadius: [v.BORDER_RADIUS, 0, 0, v.BORDER_RADIUS],
        display: 'flex',
        flexDirection: 'row',
        padding: [v.UNIT2, 0, v.UNIT2, v.UNIT2],
        position: 'absolute',
        top: v.UNIT2,
        right: v.UNIT2,

        '.modToolbarItem': {
        },

        '.modToolbarGroup': {
            display: 'flex',
            flexDirection: 'row',

            '&.isActive': {
                '.modToolbarItem': {
                    display: 'block'
                },
            },

            '&.isInactive': {
                '.modToolbarItem': {
                    display: 'none'
                },
            },

            '&.isNormal': {
                '.modToolbarItem': {
                    display: 'none'
                },
                '.modToolbarItem.isLastUsed': {
                    display: 'block'
                },
            },
        },

        '.modToolbarItem .uiIconButton': {
            borderRadius: v.BORDER_RADIUS,
            backgroundColor: v.TOOLBAR_BUTTON_BACKGROUND,
            marginLeft: v.UNIT2,
            ...v.TRANSITION('all'),
            '&.isActive': {
                backgroundColor: v.TOOLBAR_ACTIVE_BUTTON_BACKGROUND,
            },
        },

        '.uiIconButton.modToolbarCancelButton': {
            backgroundColor: v.TOOLBAR_CLOSE_BUTTON_BACKGROUND,
            ...v.LOCAL_SVG('double-arrow', v.TOOLBAR_CLOSE_BUTTON_COLOR)

        },

        '.uiSelect': {
            '.uiMenu, &.isUp .uiMenu': {
                backgroundColor: v.TOOLBAR_BACKGROUND,
                borderWidth: 0,
                borderRadius: [v.UNIT4, v.UNIT4, 0, 0],

            },
            '.uiRawInput': {
                color: v.TOOLBAR_TEXT_COLOR,
            },
            '.uiMenuItem': {
                color: v.TOOLBAR_TEXT_COLOR,
                '&:hover': {
                    backgroundColor: v.COLOR.black,
                },
            },
            '.uiControlBox': {
                borderWidth: 0,

            }
        },


    }
});
