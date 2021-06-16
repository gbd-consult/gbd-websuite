

module.exports = v => ({

    '.uiIconButton.modZoomInfobarOutButton': {
        ...v.SVG(__dirname + '/zoom_out', v.INFOBAR_ICON_COLOR),
    },

    '.uiIconButton.modZoomInfobarInButton': {
        ...v.SVG(__dirname + '/zoom_in', v.INFOBAR_ICON_COLOR),
    },

    '.uiIconButton.modZoomInfobarBoxButton': {
        ...v.SVG(__dirname + '/zoom_rectangle', v.INFOBAR_ICON_COLOR),
    },

    '.uiIconButton.modZoomInfobarResetButton': {
        ...v.SVG(__dirname + '/zoom_reset', v.INFOBAR_ICON_COLOR),
    },

    '.modZoomBox': {
        borderWidth: 2,
        borderStyle: 'dashed',
        borderColor: v.ZOOM_BOX_COLOR,
    }

});
