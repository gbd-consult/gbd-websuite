module.exports = v => ({
    '.mapshareToolbarButton': {
        ...v.TOOLBAR_BUTTON(__dirname + '/share')
    },
    '.uiDialog.mapshareDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(420, 550),
        }
    },
    '.mapshareQrCodeBox': {
        height: 200,
        display: 'flex',
        alignItems: 'center',
        'img': {
            display: 'block',
            margin: 'auto',
            width: 200,
            height: 200,
        },
    },
});
