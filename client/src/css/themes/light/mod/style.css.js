module.exports = v => ({
    '.modStyleSidebarIcon': {
        ...v.SIDEBAR_ICON('google:image/color_lens')
    },

    '.modStyleSidebar': {
        '.modSidebarTabHeader': {
            padding: [v.UNIT, v.UNIT4, v.UNIT, v.UNIT2],
            '.uiControlBox': {
                border: 'none',
                xbackgroundColor: v.EVEN_STRIPE_COLOR,
            },
            '.uiRawInput': {
                textAlign: 'right',
            }
        }
    },
    '.modStyleRenameControl': {
        '.uiInput': {
            flex: 1,
        },
        '.uiIconButton': {
            ...v.ICON_BUTTON(),
            ...v.ICON_SIZE('small'),
            ...v.SVG('google:communication/swap_calls', v.BORDER_COLOR),
        }
    }


});
