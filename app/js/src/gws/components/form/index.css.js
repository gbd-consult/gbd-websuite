module.exports = v => ({
    '.cmpForm': {

        borderCollapse: 'collapse',
        width: '100%',

        'tr.isError': {
            '.uiControlBox': {
                borderColor: v.ERROR_COLOR,
            }
        },

        'td, th': {
            verticalAlign: 'middle',
            padding: [v.UNIT2, 0, v.UNIT2, 0],
            //fontSize: v.SMALL_FONT_SIZE,
            //borderWidth: 1,
            borderStyle: 'dotted',
            borderColor: v.BORDER_COLOR,
            textAlign: 'left',
            lineHeight: '120%',
        },

        'td': {
            maxWidth: 300,

        },
        'th': {
            fontWeight: 'bold',
            paddingRight: v.UNIT2,
            maxWidth: 100,

        },

        'tr.cmpFormError': {
            'td, th': {
                fontSize: v.TINY_FONT_SIZE,
                color: v.ERROR_COLOR,
                opacity: 0,
                padding: 0,
                ...v.TRANSITION('opacity'),
            },

            '&.isActive': {
                'td, th': {
                    opacity: 1,
                    padding: [v.UNIT, 0, v.UNIT, 0],
                ...v.TRANSITION('opacity'),
                }
            }

        },

    },

    '.cmpFormList': {
        border: [1, 'solid', v.BORDER_COLOR],
        position: 'relative',

        '.cmpList': {
            maxHeight: v.UNIT * 40,
            overflow: 'auto',
        },

        '.cmpFormListToolbar': {
            backgroundColor: v.SIDEBAR_AUX_TOOLBAR_BACKGROUND,

            '.uiIconButton': {
                ...v.ICON_SIZE('small'),
                '&.isDisabled': {
                    opacity: 0.5,
                },

            },
        },

        '.cmpFormListNewButton': {...v.SVG('google:content/add_circle_outline', v.SIDEBAR_AUX_BUTTON_COLOR)},
        '.cmpFormListEditButton': {...v.SVG('google:image/edit', v.SIDEBAR_AUX_BUTTON_COLOR)},
        '.cmpFormListDeleteButton': {...v.SVG('google:action/delete_forever', v.SIDEBAR_AUX_BUTTON_COLOR)},
        '.cmpFormListLinkButton': {...v.SVG('google:content/link', v.SIDEBAR_AUX_BUTTON_COLOR)},
        '.cmpFormListUnlinkButton': {...v.SVG(__dirname + '/link_off_black_24dp', v.SIDEBAR_AUX_BUTTON_COLOR)},

        '.cmpFormFileViewButton': {...v.SVG('google:action/visibility', v.SIDEBAR_AUX_BUTTON_COLOR)},

    },

    '.cmpFormDrawGeometryButton': {
        border: [1, 'solid', v.BORDER_COLOR],
        ...v.ICON_SIZE('medium'),
        ...v.SVG(__dirname + '/draw_black_24dp', v.TEXT_COLOR),
        '&.isActive': {
            ...v.SVG(__dirname + '/draw_black_24dp', v.FOCUS_COLOR),
        }
    },

    '.cmpFormEditGeometryButton': {
        border: [1, 'solid', v.BORDER_COLOR],
        ...v.ICON_SIZE('medium'),
        ...v.SVG(__dirname + '/cursor', v.TEXT_COLOR),
    },





});