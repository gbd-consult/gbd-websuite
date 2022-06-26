module.exports = v => ({

    '.modEditToolbarButton': {
        ...v.TOOLBAR_BUTTON('google:image/edit')
    },

    '.modEditSelected': {
        marker: 'circle',
        markerStroke: v.COLOR.opacity(v.COLOR.pink800, 0.8),
        markerStrokeWidth: 4,
        markerSize: 25,
        markerStrokeDasharray: '4',
    },

    '.modEditSidebar': {
        '.modSidebarTabHeader': {
            padding: v.UNIT4,
        },
        '.modSidebarTabBody': {
            padding: 0,
            '.uiVRow': {
                padding: [v.UNIT2, 0, 0, v.UNIT2],
            }

        },
        '&.modEditSidebarFormTab .modSidebarTabBody .uiVRow': {
            padding: [v.UNIT4],
        },
        '.modSearchBox': {
            padding: 0,
            // borderBottom: [1, 'solid', v.BORDER_COLOR],
            // backgroundColor:v.SIDEBAR_AUX_TOOLBAR_BACKGROUND,

        },

    },

    '.modEditSidebarIcon': {
        ...v.SIDEBAR_ICON('google:image/edit')
    },

    '.modEditorLayerListButton': {
        ...v.LIST_BUTTON('google:image/edit')
    },

    '.modEditPointerAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('cursor')
    },
    '.modEditGotoLayersAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:maps/layers')
    },

    '.modEditGotoFeaturesAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:action/reorder')
    },

    '.modEditDrawAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('draw_black_24dp')
    },
    '.modEditAddAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:content/add_circle_outline')
    },
    '.modEditRemoveAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:action/delete')
    },

    '.modEditEndButton': {
        ...v.SVG('google:action/done')
    },
    '.modEditSaveButton': {
        ...v.ROUND_FORM_BUTTON('google:content/save'),
        backgroundColor: v.PRIMARY_BACKGROUND,
        opacity: 0.3,
        '&.isActive': {
            opacity: 1,
        }

    },
    '.modEditCancelButton': {
        ...v.ROUND_CLOSE_BUTTON(),
    },
    '.modEditDeleteButton': {
        ...v.ROUND_FORM_BUTTON('google:action/delete'),
        backgroundColor: v.COLOR.red300,
    },
    '.modEditResetButton': {
        ...v.ROUND_FORM_BUTTON('google:content/undo'),
        opacity: 0.3,
        '&.isActive': {
            opacity: 1,
        }
    },

    '.modEditStyleButton': {
        ...v.ROUND_FORM_BUTTON('google:image/brush')
    },

    '.modEditSelectFeatureDialog .uiDialogContent': {
        padding: 0,
        '.uiVBox': {
            height: 350,
            padding: [0, 0, 0, v.UNIT4],

        }
    }


});