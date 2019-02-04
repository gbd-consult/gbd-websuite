module.exports = v => ({

    '.modSidebar': {
        left: '-150%',
        width: '100%',
        bottom: v.INFOBAR_HEIGHT,

        [v.MEDIA('small+')]: {
            left: -v.SIDEBAR_WIDTH,
            width: v.SIDEBAR_WIDTH,
        }
    },

    '.modSidebarOverflowPopup': {
        right: v.UNIT4,
        [v.MEDIA('small+')]: {
            left: v.SIDEBAR_WIDTH - v.SIDEBAR_POPUP_WIDTH - v.UNIT4,
            right: 'auto',
        }
    },

    '.modInfobarWidget': {
        '&.modInfobarRotation, &.modInfobarPosition, &.modInfobarScale': {
            display: 'none',
            [v.MEDIA('large+')]: {
                display: 'flex'
            }
        }
    },

    '.modPrintPreviewDialog': {
        left: 0,
        top: 0,
        right: 0,
        [v.MEDIA('small+')]: {
            left: 'auto',
            width: 350,
            right: v.UNIT4,
            top: v.UNIT4,
        }
    },

    '.modDrawControlBox.isActive': {
        left: 0,
        top: 0,
        right: 0,
        [v.MEDIA('small+')]: {
            left: 'auto',
            right: v.UNIT4,
            top: v.UNIT4,
        }
    },

    '.modInfobox': {
        left: 0,
        right: 0,
        bottom: '-100%',
        [v.MEDIA('small+')]: {
            left: 'auto',
            // @TODO accomodate with the search popup
            right: v.UNIT8,
        }
    },

    '.modInfobox.isActive': {
        bottom: v.INFOBAR_HEIGHT,
        // [v.MEDIA('small+')]: {
        //     bottom: v.INFOBAR_HEIGHT + v.UNIT * 6,
        // }
    },

    '.uiDialog': {
        left: 0,
        top: 0,
        width: '100%',
        height: '100%',

        [v.MEDIA('large+')]: {
            left: '50%',
            top: '50%',
            margin: 'auto',
            ...v.SHADOW,
            ...v.CENTER_BOX(800, 600),

            '&.modPrintProgressDialog': {
                ...v.CENTER_BOX(400, 190),
            },
            '&.modGekosDialog': {
                ...v.CENTER_BOX(300, 280),
            },
            '&.modSelectDialog': {
                ...v.CENTER_BOX(300, 200),
            },
            '&.modAlkisSelectDialog': {
                ...v.CENTER_BOX(300, 200),
            }
        },
    },

    '.modDecorationScaleRuler': {
        display: 'none',
        [v.MEDIA('medium+')]: {
            display: 'block',
        }
    }

});
