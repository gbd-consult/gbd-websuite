module.exports = v => ({

    '.modStorageWriteAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:content/save')
    },

    '.modStorageReadAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:file/folder_open')
    },

    '.uiDialog.modStorageReadDialog': {
        '.uiListBox': {
            height: 260,

        },
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(500, 430),
        },
    },

    '.uiDialog.modStorageWriteDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(500, 490),
        },
    },

    '.modStorageDeleteButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('tiny'),
        ...v.SVG('google:action/delete_forever', v.BORDER_COLOR),
    },

});
