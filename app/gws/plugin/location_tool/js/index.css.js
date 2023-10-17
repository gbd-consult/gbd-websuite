module.exports = v => ({
    '.locationToolToolbarButton': {
        ...v.TOOLBAR_BUTTON('google:maps/my_location'),
    },
    '.uiDialog.locationToolErrorDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(500, 200),
        },
        '.uiTextBlock': {
            padding: v.UNIT4,
            textAlign: 'center',
        }
    }
});
