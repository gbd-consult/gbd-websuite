module.exports = v => ({

    '.modStorageWriteAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:content/save')
    },

    '.modStorageReadAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:file/folder_open')
    },

    '.uiDialog.modStorageReadDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(300, 500),
        },
    },

    '.uiDialog.modStorageWriteDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(300, 300),
        },
    }


});
