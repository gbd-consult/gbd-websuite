module.exports = v => ({
    '.modGeorisksToolbarButton': {
        ...v.TOOLBAR_BUTTON('outline-report_problem-24px')
    },
    '.modGeorisksFormPadding': {
        textAlign: 'center',
        padding: [v.UNIT8],
    },

    '.uiDialog.modGeorisksDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(600, 700),
        },
        '.cmpPropertySheet th': {
            width: 150,
        },
        '.cmpPropertySheet td b': {
            fontSize: v.SMALL_FONT_SIZE,
            paddingLeft: v.UNIT2,
        },

        '.uiToggle': {
            display: 'inline-block',
        }
    },

    '.uiDialog.modGeorisksSmallDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(500, 200),
        },
    }
});
