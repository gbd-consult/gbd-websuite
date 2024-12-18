module.exports = v => ({

    '.uiShowPasswordButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('tiny'),
        ...v.SVG(__dirname + '/visibility_24dp_0_FILL0_wght400_GRAD0_opsz24', v.BORDER_COLOR),
        '&.isOpen': {
            ...v.SVG(__dirname + '/visibility_off_24dp_0_FILL0_wght400_GRAD0_opsz24', v.BORDER_COLOR),
        }
    },
});
