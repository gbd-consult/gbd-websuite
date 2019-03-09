module.exports = v => ({
    '.modLayersSidebar': {
        ...v.SIDEBAR_ICON('google:maps/layers')
    },

    '.modLayersTreeRow': {
        ...v.TRANSITION('background-color'),

        '&:hover': {
            backgroundColor: v.HOVER_COLOR,
        },

        '.modLayersTreeTitle': {
            opacity: 0.6,
            lineHeight: 1.3,
            overflow: 'hidden',
            '.uiRawButton': {
                textAlign: 'left',
            }
        },

        '.uiIconButton': {
            opacity: 0.4,
            ...v.ICON_SIZE('small'),
            ...v.TRANSITION(),
        },

        '&.isSelected': {
            '.modLayersTreeTitle': {
                color: v.HIGHLIGHT_COLOR,
            },
        },

        '&.isVisible': {
            '.modLayersTreeTitle': {
                opacity: 1,
            },
            '.uiIconButton': {
                opacity: 1,
            }
        },
    },

    '.modLayersTreeChildren': {
        paddingLeft: 20
    },

    '.modLayersExpandButton': {
        ...v.SVG('google:navigation/chevron_right', v.TEXT_COLOR),
    },

    '.modLayersCollapseButton': {
        ...v.SVG('google:navigation/chevron_right', v.TEXT_COLOR),
        transform: 'rotate(90deg)',
    },

    '.modLayersLeafButton': {
        ...v.SVG('google:image/crop_7_5', v.TEXT_COLOR),
    },

    '.modLayersCheckButton': {
        ...v.SVG('google:action/visibility_off', v.TEXT_COLOR),
    } ,

    '.modLayersCheckButton.isChecked': {
        ...v.SVG('google:action/visibility', v.TEXT_COLOR),
    } ,

    '.modLayersCheckButton.isExclusive': {
        ...v.SVG('google:toggle/radio_button_unchecked', v.TEXT_COLOR),
    } ,

    '.modLayersCheckButton.isExclusive.isChecked': {
        ...v.SVG('google:toggle/radio_button_checked', v.TEXT_COLOR),
    } ,

    '.modLayersCheckButton.isVisible': {
        ...v.SVG('google:action/visibility_off', v.FOCUS_COLOR),
    } ,

    '.modLayersCheckButton.isChecked.isVisible': {
        ...v.SVG('google:action/visibility', v.FOCUS_COLOR),
    } ,

    '.modLayersCheckButton.isExclusive.isVisible': {
        ...v.SVG('google:toggle/radio_button_unchecked', v.FOCUS_COLOR),
    } ,

    '.modLayersCheckButton.isExclusive.isChecked.isVisible': {
        ...v.SVG('google:toggle/radio_button_checked', v.FOCUS_COLOR),
    } ,

    '.modLayersDetails': {
        borderTop: [1, 'solid', v.BORDER_COLOR],
    },

    '.modLayersDetailsBody': {
        maxHeight: 400,
        overflow: 'auto',
    },

    '.modLayersDetailsBodyContent': {
        padding: v.UNIT4,
    },

    '.modLayersZoomAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('zoom_layer')
    },
    '.modLayersShowAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:maps/layers_clear')
    },
    '.modLayersEditAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:image/edit')
    },


});
