module.exports = v => ({
    '.modAlkisSidebarIcon': {
        ...v.GOOGLE_SVG('communication/business', v.SIDEBAR_HEADER_COLOR)
    },

    '.modAlkisLensToolbar': {
        marginTop: 40,
        borderTopWidth: 1,
        borderStyle: 'solid',
        borderColor: v.BORDER_COLOR,

        '.uiIconButton': {
            ...v.ICON('normal'),
        },
    },

    '.modAlkisExportButton': {
        ...v.LOCAL_SVG('download'),
    },

    '.uiIconButton.modAlkisFeatureSelectIcon': {
        ...v.ICON('small'),
        ...v.GOOGLE_SVG('content/add_circle_outline', v.FOCUS_COLOR),
    },

    '.uiIconButton.modAlkisFeatureUnselectIcon': {
        ...v.ICON('small'),
        ...v.GOOGLE_SVG('content/remove_circle_outline', v.FOCUS_COLOR),
    },

    '.modAlkisHighlightButton': {
        ...v.ICON('small'),
        ...v.GOOGLE_SVG('image/center_focus_weak'),
    },

    '.modAlkisSelectButton': {
        ...v.ICON('small'),
        ...v.GOOGLE_SVG('content/add_circle_outline'),
    },

    '.modAlkisSelectAllButton': {
        ...v.ICON('small'),
        ...v.GOOGLE_SVG('content/add_circle_outline'),
    },

    '.modAlkisUnselectButton': {
        ...v.ICON('small'),
        ...v.GOOGLE_SVG('content/remove_circle_outline'),
    },

    '.modAlkisSelectionButton': {
        ...v.ICON('small'),
        ...v.GOOGLE_SVG('action/bookmark_border'),
    },

    '.modAlkisSelectionButtonDisabled': {
        ...v.ICON('small'),
        ...v.GOOGLE_SVG('editor/format_align_justify', v.BORDER_COLOR),
    },

    '.modAlkisSelectionButtonText': {
        fontSize: v.TINY_FONT_SIZE
    },

    '.modAlkisClearSelectionButton': {
        ...v.ICON('small'),
        ...v.GOOGLE_SVG('action/delete_forever'),
    },

    '.modAlkisEmptyTab': {
        textAlign: 'center',
        padding:
            30,
        lineHeight:
            1.3,
        color:
        v.BORDER_COLOR,

        'a':
            {
                display: 'block',
                marginTop:
                    30,
                color:
                v.FOCUS_COLOR,
                cursor:
                    'pointer',
            }

    }

});