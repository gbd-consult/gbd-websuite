module.exports = v => ({
    '.modFsinfoSidebarIcon': {
        ...v.ROUND_FORM_BUTTON('person_search-24px')
    },

    '.modFsinfoSearchSubmitButton': {
        ...v.ROUND_FORM_BUTTON(v.SEARCH_ICON)
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
            ...v.CENTER_BOX(400, 400),
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

    '.modFsinfoCreateDocumentButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('medium'),
        ...v.SVG('google:content/add_circle_outline', v.BUTTON_COLOR),
    },


});
