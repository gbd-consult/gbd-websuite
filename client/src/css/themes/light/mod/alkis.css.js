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

    '.modAlkisLoading': {
        textAlign: 'center',
        color: v.DISABLED_COLOR,
        padding: [v.UNIT2 * 8, v.UNIT4, v.UNIT4, v.UNIT4],
        lineHeight: 1.3,
    },


    '.modAlkisSubmitButton': {
        ...v.GOOGLE_SVG('action/search', v.BUTTON_COLOR),
        backgroundColor: v.BUTTON_BACKGROUND,
        borderRadius: v.BORDER_RADIUS,
        '&.isActive': {
            backgroundColor: v.PRIMARY_BACKGROUND,


        }

    },

    '.modAlkisLensButton': {
        ...v.LOCAL_SVG('search_lens', v.BUTTON_COLOR),
        backgroundColor: v.BUTTON_BACKGROUND,
        borderRadius: v.BORDER_RADIUS,
        '&.isActive': {
            backgroundColor: v.PRIMARY_BACKGROUND,
        }
    },


    '.uiIconButton.modAlkisFeatureSelectIcon': {
        ...v.ICON('small'),
        ...v.GOOGLE_SVG('content/add_circle_outline', v.FOCUS_COLOR),
    },

    '.uiIconButton.modAlkisFeatureUnselectIcon': {
        ...v.ICON('small'),
        ...v.GOOGLE_SVG('content/remove_circle_outline', v.FOCUS_COLOR),
    },

    '.modAlkisExportButton': {
        ...v.LOCAL_SVG('baseline-save_alt-24px', v.SECONDARY_BUTTON_COLOR),
    },


    '.modAlkisHighlightButton': {
        ...v.GOOGLE_SVG('image/center_focus_weak', v.SECONDARY_BUTTON_COLOR),
    },

    '.modAlkisSelectButton': {
        ...v.GOOGLE_SVG('content/add_circle_outline', v.SECONDARY_BUTTON_COLOR),
    },

    '.modAlkisPrintButton': {
        ...v.GOOGLE_SVG('action/print', v.SECONDARY_BUTTON_COLOR),
    },

    '.modAlkisSelectAllButton': {
        ...v.GOOGLE_SVG('content/add_circle_outline', v.SECONDARY_BUTTON_COLOR),
    },

    '.modAlkisUnselectButton': {
        ...v.GOOGLE_SVG('content/remove_circle_outline', v.SECONDARY_BUTTON_COLOR),
    },

    '.modAlkisGotoSelectionButton': {
        ...v.GOOGLE_SVG('action/bookmark_border', v.SECONDARY_BUTTON_COLOR),
    },

    '.modAlkisGotoFormButton': {
        ...v.GOOGLE_SVG('communication/business', v.SECONDARY_BUTTON_COLOR),
    },

    '.modAlkisGotoListButton': {
        ...v.GOOGLE_SVG('action/reorder', v.SECONDARY_BUTTON_COLOR),
    },

    '.modAlkisClearSelectionButton': {
        ...v.GOOGLE_SVG('action/delete_forever', v.SECONDARY_BUTTON_COLOR),
    },


    '.modAlkisSelectionButtonDisabled': {
        ...v.ICON('small'),
        ...v.GOOGLE_SVG('editor/format_align_justify', v.BORDER_COLOR),
    },

    '.modAlkisSelectionButtonText': {
        fontSize: v.TINY_FONT_SIZE
    },


});