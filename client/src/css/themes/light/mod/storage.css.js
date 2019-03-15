module.exports = v => ({

    '.modStorageWriteAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:content/save')
    },

    '.modStorageReadAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:file/folder_open')
    },

    '.uiDialog.modStorageDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(300, 250),
        },
    }


});
