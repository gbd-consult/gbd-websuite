module.exports = v => ({
    '.modFsinfoSidebarIcon': {
        ...v.ROUND_FORM_BUTTON('person_search-24px')
    },

    '.modFsinfoSearchSubmitButton': {
        ...v.ROUND_FORM_BUTTON(v.SEARCH_ICON)
    },

    '.modFsinfoSearchResetButton': {
        ...v.ROUND_FORM_BUTTON('google:content/delete_sweep')
    },

    '.modFsinfoLoading': {
        textAlign: 'center',
        color: v.DISABLED_COLOR,
        padding: [v.UNIT2 * 8, v.UNIT4, v.UNIT4, v.UNIT4],
        lineHeight: 1.3,
    },

    '.modFsinfoPerson': {
        // paddingBottom: v.UNIT4,
        // marginBottom: v.UNIT4,

        '.head2': {
            padding: v.UNIT,
            backgroundColor: v.EVEN_STRIPE_COLOR,
        }
    },

    '.uiDialog.modFsinfoUploadDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(800, 500),
        },


    },

    '.uiDialog.modFsinfoDeleteDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(300, 300),
        },


    },


    '.modFsinfoSearchAuxButton': {...v.SIDEBAR_AUX_BUTTON(v.SEARCH_ICON)},
    '.modFsinfoDetailsAuxButton': {...v.SIDEBAR_AUX_BUTTON('contact_page-24px')},
    '.modFsinfoDocumentsAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:action/reorder')},


    '.modFsinfoViewDocumentButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG('google:action/visibility', v.BUTTON_COLOR),
    },

    '.modFsinfoDeleteDocumentButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG('google:action/delete', v.BUTTON_COLOR),
    },


    '.modFsinfoUpdateDocumentButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG('google:editor/publish', v.BUTTON_COLOR),
    },

    '.modFsinfoUploadButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG('google:content/add_circle_outline', v.BUTTON_COLOR),
    },

    '.modFsinfoDownloadButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG('google:action/get_app', v.BUTTON_COLOR),
    },

    '.modFsinfoFileList': {
        height: 240,
        overflow: 'auto',

        '.uiRow': {
            marginTop: v.UNIT4,
        }

    },

    '.modFsinfoDeleteListButton': {
        ...v.LIST_BUTTON('google:action/delete_forever')
    },

    '.uiIconButton.modFsinfoExpandButton': {
        ...v.SVG('google:navigation/chevron_right', v.TEXT_COLOR),
        ...v.ICON_SIZE('small'),
        ...v.TRANSITION('transform'),

    },


    '.modFsinfoRecord.isOpen ': {
        '.modFsinfoExpandButton': {
            transform: 'rotate(90deg)',
        },
        '.modFsinfoRecordContent': {
            display: 'block'
        },
    },

    '.modFsinfoRecordHead': {
        backgroundColor: v.COLOR.blueGrey50,
        paddingLeft: v.UNIT4,
        marginBottom: v.UNIT,


    },

    '.modFsinfoRecordContent': {
        display: 'none',

    },

    '.modFsinfoDocumentList': {
        '.uiRow': {
            backgroundColor: v.EVEN_STRIPE_COLOR,
            marginBottom: v.UNIT,
        },
    },

    '.modFsinfoDocumentName': {
        fontSize: v.SMALL_FONT_SIZE,
        padding: v.UNIT4,
    },

    '.modFsinfoDialogMessage': {
        fontWeight: 800,
        padding: v.UNIT4,
    }


});
