module.exports = v => ({
    '.modLocationToolbarButton': {
        ...v.TOOLBAR_BUTTON('google:maps/my_location'),
    },
    '.uiDialog.modLocationErrorDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(500, 200),
        },
        '.uiTextBlock': {
            padding: v.UNIT4,
            textAlign: 'center',
        }
    }
});
