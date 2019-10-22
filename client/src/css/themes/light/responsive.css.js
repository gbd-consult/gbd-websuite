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

    '.cmpInfobox': {
        left: 0,
        right: 0,
        bottom: '-100%',
        [v.MEDIA('small+')]: {
            left: 'auto',
            // @TODO accomodate with the search popup
            right: v.UNIT8,
        }
    },

    '.cmpInfobox.isActive': {
        bottom: v.INFOBAR_HEIGHT,
        // [v.MEDIA('small+')]: {
        //     bottom: v.INFOBAR_HEIGHT + v.UNIT * 6,
        // }
    },


    '.modDecorationScaleRuler': {
        display: 'none',
        [v.MEDIA('medium+')]: {
            display: 'block',
        }
    }

});
