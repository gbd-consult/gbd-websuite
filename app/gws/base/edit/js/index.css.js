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

        '.editSidebar': {
            '.modSidebarTabHeader': {
                padding: v.UNIT4,
            },
            '.modSidebarTabBody': {
                padding: 0,
                '.uiVRow': {
                    padding: [v.UNIT2, 0, 0, v.UNIT2],
                    marginBottom: 0,
                }

            },
            '&.editSidebarFormTab .modSidebarTabBody .uiVRow': {
                padding: [v.UNIT4],
            },
            '.modSearchBox': {
                padding: 0,
                '.uiControlBox': {
                    border: 0
                },
                // backgroundColor:v.SIDEBAR_AUX_TOOLBAR_BACKGROUND,
            },
        },


        '.editSidebarIcon': {
            ...v.SIDEBAR_ICON('google:image/edit')
        },

        '.editToolbarButton': {
            ...v.TOOLBAR_BUTTON('google:image/edit')
        },


        '.editModelButton': {
            ...v.LIST_BUTTON(__dirname + '/database_FILL1_wght400_GRAD0_opsz24')
        },

        '.editTableViewButton': {
            ...v.LIST_BUTTON(__dirname + '/table_FILL1_wght400_GRAD0_opsz24')
        },

        '.editModifyAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON(__dirname + '/cursor')
        },

        '.editEndButton': {
            ...v.SVG('google:action/done')
        },
        '.editSaveButton': {
            ...v.ROUND_OK_BUTTON(),
            opacity: 0.3,
            '&.isActive': {
                opacity: 1,
            }
        },
        '.editCancelButton': {
            ...v.ROUND_CLOSE_BUTTON(),
        },
        '.editResetButton': {
            ...v.ROUND_FORM_BUTTON('google:content/undo'),
            opacity: 0.3,
            '&.isActive': {
                opacity: 1,
            }
        },
        '.editDeleteButton': {
            ...v.ROUND_FORM_BUTTON('google:action/delete'),
            backgroundColor: v.COLOR.pink200,

        },
        '.editOpenFormButton': {
            ...v.ROUND_FORM_BUTTON(__dirname + '/subject_FILL1_wght400_GRAD0_opsz24'),

        },

        '.editNewButton': {
            ...v.ROUND_FORM_BUTTON('google:content/add_circle_outline'),

        },

        '.editStyleButton': {
            ...v.ROUND_FORM_BUTTON('google:image/brush')
        },


        '.editModelListAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON(__dirname + '/database_FILL1_wght400_GRAD0_opsz24')
        },

        '.editTableViewAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON(__dirname + '/table_FILL1_wght400_GRAD0_opsz24')
        },

        '.editCloseFeatureAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/reorder')
        },

        '.editDrawAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON(__dirname + '/draw_black_24dp')
        },
        '.editNewAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:content/add_circle_outline')
        },
        '.editRemoveAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/delete')
        },


        '.editTableError': {
            fontSize: v.TINY_FONT_SIZE,
            color: v.ERROR_COLOR,
            opacity: 0,
            padding: 0,
            ...v.TRANSITION('opacity'),

            '&.isActive': {
                opacity: 1,
                padding: [v.UNIT, v.UNIT2],
                ...v.TRANSITION('opacity'),
            }

        },


        // dialogs

        '.uiDialog.editSelectFeatureDialog': {
            [v.MEDIA('large+')]: {
                ...v.CENTER_BOX(400, 600),
            },
            '.uiControlBox': {
                border: 0
            },
        },

        '.uiDialog.editTableViewDialog': {
            [v.MEDIA('large+')]: {
                ...v.BIG_BOX(),
            },
        },



        // geometry styles

        '.editFeature.point, .editFeature.multipoint': {
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

        '.editFeature.linestring, .editFeature.multilinestring': {
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

        '.editFeature.polygon, .editFeature.multipolygon': {
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