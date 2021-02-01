module.exports = v => ({
    '.modTabeditSidebarIcon': {
        ...v.SIDEBAR_ICON('table_view-24px')
    },
    '.uiDialog.modTabeditDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(720, 700),
            '&.isZoomed': {
                ...v.FIT_SCREEN()
            }
        },
    },

    '.uiDialog.modTabeditSmallDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(500, 200),
        },
    },

    '.modTabeditListButton': {
        ...v.SVG('google:image/crop_7_5', v.TEXT_COLOR),
    },

    '.modTabeditListTitle': {
        opacity: 0.6,
        lineHeight: 1.3,
        overflow: 'hidden',
        '.uiRawButton': {
            textAlign: 'left',
        }
    },


    '.modTabeditDialogLoading': {
        textAlign: 'center',
        padding: v.UNIT8,
    },

    '.uiIconButton.modTabeditButtonSave': {
        ...v.ICON_SIZE('normal'),
        ...v.SVG('google:content/save', v.PRIMARY_COLOR),
        backgroundColor: v.PRIMARY_BACKGROUND,
        borderRadius: v.BORDER_RADIUS,

        '&.isDisabled': {
            opacity: 0.3
        }

    },

    '.uiIconButton.modTabeditButtonAdd': {
        ...v.ICON_SIZE('normal'),
        ...v.SVG('google:content/add_circle_outline', v.PRIMARY_COLOR),
        backgroundColor: v.PRIMARY_BACKGROUND,
        borderRadius: v.BORDER_RADIUS,

        '&.isDisabled': {
            opacity: 0.3
        }

    },

    '.modTabeditFilter .uiIconButton': {
        ...v.ICON_SIZE('tiny'),
        ...v.SVG(v.SEARCH_ICON, v.BORDER_COLOR),
    },



});
