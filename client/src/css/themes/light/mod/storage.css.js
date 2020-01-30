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
            ...v.CENTER_BOX(300, 430),
        },
    },

    '.uiDialog.modStorageWriteDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(300, 490),
        },
    }


});
