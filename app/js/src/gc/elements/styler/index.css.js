module.exports = v => {

    let ICON_RADIO = (prop, icon) => {
        prop = '.uiToggle.stylerProp_' + prop

        return {
            [prop + ' button']: {
                ...v.SVG(icon, v.TEXT_COLOR)
            },
            [prop + '.isChecked button']: {
                // ...v.SVG(icon, v.COLOR.blueGrey50),
                // backgroundColor: v.COLOR.blueGrey700,
                border: [3, 'solid', v.BORDER_COLOR],
                borderRadius: v.BORDER_RADIUS,
            },
        }
    }


    return {
        '.stylerSidebarIcon': {
            ...v.SIDEBAR_ICON('google:image/color_lens')
        },

        '.stylerSidebar': {
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
        '.stylerRenameControl': {
            '.uiInput': {
                flex: 1,
            },
            '.uiIconButton': {
                ...v.ICON_BUTTON(),
                ...v.ICON_SIZE('small'),
                ...v.SVG('google:communication/swap_calls', v.BORDER_COLOR),
            }
        },

        ...ICON_RADIO('label_align_left', 'google:editor/format_align_left'),
        ...ICON_RADIO('label_align_center', 'google:editor/format_align_center'),
        ...ICON_RADIO('label_align_right', 'google:editor/format_align_right'),
        ...ICON_RADIO('label_placement_start', __dirname + '/placement_start'),
        ...ICON_RADIO('label_placement_middle', __dirname + '/placement_middle'),
        ...ICON_RADIO('label_placement_end', __dirname + '/placement_end'),
    }
};
