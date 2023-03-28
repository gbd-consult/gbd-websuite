module.exports = v => {
    let DARK = v.COLOR.cyan500;
    let LIGHT = v.COLOR.cyan50;
    let FOCUS = v.COLOR.pink500;
    let STROKE = 6;

    let MARKER = {
        markerType: 'circle',
        markerFill: LIGHT,
        markerStroke: FOCUS,
        markerStrokeWidth: 3,
        markerSize: 15,
    }

    let LABEL = {
        withLabel: 'all',
        labelFontSize: 12,
        labelFill: DARK,
        labelStroke: v.COLOR.white,
        labelStrokeWidth: 6,
    };


    return {

        '.modEditSidebarIcon': {
            ...v.SIDEBAR_ICON('google:image/edit')
        },

        '.modEditToolbarButton': {
            ...v.TOOLBAR_BUTTON('google:image/edit')
        },


        '.modEditorModelListButton': {
            ...v.LIST_BUTTON('google:image/edit')
        },

        '.modEditModifyAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON(__dirname + '/cursor')
        },

        '.modEditEndButton': {
            ...v.SVG('google:action/done')
        },
        '.modEditSaveButton': {
            ...v.ROUND_OK_BUTTON(),
        },
        '.modEditCancelButton': {
            ...v.ROUND_CLOSE_BUTTON(),
        },
        '.modEditResetButton': {
            ...v.ROUND_FORM_BUTTON('google:content/undo'),
            opacity: 0.3,
            '&.isActive': {
                opacity: 1,
            }
        },
        '.modEditDeleteButton': {
            ...v.ROUND_FORM_BUTTON('google:action/delete')
        },

        '.modEditStyleButton': {
            ...v.ROUND_FORM_BUTTON('google:image/brush')
        },


        '.modEditGotoModelListAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:maps/layers')
        },

        '.modEditCloseFeatureAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/reorder')
        },

        '.modEditDrawAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON(__dirname + '/draw_black_24dp')
        },
        '.modEditAddAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:content/add_circle_outline')
        },
        '.modEditRemoveAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/delete')
        },


        // geometry styles

        '.modEditFeature.point, .modEditFeature.multipoint': {
            fill: v.COLOR.opacity(DARK, 0.7),
            stroke: LIGHT,
            strokeWidth: STROKE,
            pointSize: 15,
            ...LABEL,
            '&.isFocused': {
                fill: v.COLOR.opacity(FOCUS, 0.7),
                strokeWidth: STROKE,
                pointSize: 15,
                ...LABEL,
                ...MARKER,
                labelFill: FOCUS,
            }
        },

        '.modEditFeature.linestring, .modEditFeature.multilinestring': {
            stroke: DARK,
            strokeWidth: STROKE,
            ...LABEL,
            '&.isFocused': {
                stroke: FOCUS,
                strokeWidth: STROKE,
                ...MARKER,
                ...LABEL,
                labelFill: FOCUS,
            }
        },

        '.modEditFeature.polygon, .modEditFeature.multipolygon': {
            fill: v.COLOR.opacity(DARK, 0.3),
            stroke: LIGHT,
            strokeWidth: STROKE,
            ...LABEL,
            '&.isFocused': {
                fill: v.COLOR.opacity(FOCUS, 0.3),
                stroke: FOCUS,
                strokeWidth: STROKE,
                ...MARKER,
                ...LABEL,
                labelFill: FOCUS,
            }
        },


    }
}