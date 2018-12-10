module.exports = v => ({
    '.modLayersSidebarIcon': {
        ...v.GOOGLE_SVG('maps/layers', v.SIDEBAR_HEADER_COLOR)
    },

    '.modLayersChildren': {
        paddingLeft: 20
    },


    '.modLayersLayer': {
        ...v.TRANSITION('background-color'),

        '&:hover': {
            backgroundColor: v.HOVER_COLOR,
        },

        '.modLayersLayerTitle': {
            opacity: 0.6,
        },

        '.uiIconButton': {
            opacity: 0.6,
            ...v.ICON('small'),
            ...v.TRANSITION('all'),
        }
    },

    '.modLayersExpandButton': {
        ...v.GOOGLE_SVG('navigation/chevron_right', v.TEXT_COLOR),
    },

    '.modLayersCollapseButton': {
        ...v.GOOGLE_SVG('navigation/chevron_right', v.TEXT_COLOR),
        transform: 'rotate(90deg)',
    },

    '.modLayersLayerButton': {
        ...v.GOOGLE_SVG('image/crop_7_5', v.TEXT_COLOR),
    },

    '.modLayersHideButton': {
        ...v.GOOGLE_SVG('action/visibility', v.FOCUS_COLOR),
    },

    '.modLayersShowButton': {
        ...v.GOOGLE_SVG('action/visibility_off', v.FOCUS_COLOR),
    },

    '.modLayersLayer.visible': {
        '.modLayersLayerTitle': {
            opacity: 1,
        },
        '.uiIconButton': {
            opacity: 1,
            ...v.TRANSITION('all'),
        }
    },

    '.modLayersLayer.isSelected': {
        '.uiItemButton': {
            color: v.HIGHLIGHT_COLOR,
        },
    },

    '.modLayersDetailsBody': {
        maxHeight: 400,
        overflow: 'auto',
    },

    '.modLayersDetailsBodyContent': {
        padding: v.UNIT4,
    },


    '.modLayersDetailsZoomButton': {
        ...v.LOCAL_SVG('baseline-zoom_out_map-24px', v.TEXT_COLOR),

    },
    '.modLayersDetailsShowButton': {
        ...v.GOOGLE_SVG('action/visibility_off', v.TEXT_COLOR),

    },
    '.modLayersDetailsEditButton': {
        ...v.GOOGLE_SVG('image/edit', v.TEXT_COLOR)

    },
    '.modLayersDetailsCloseButton': {
        ...v.GOOGLE_SVG('navigation/close', v.TEXT_COLOR),

    },

});
