module.exports = v => ({
    '.uiDialog.accountDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(400, 600),
        },
    },

    '.accountadminSidebarIcon': {
        ...v.SIDEBAR_ICON(__dirname + '/manage_accounts')
    },
    '.accountResetButton': {
        ...v.ROUND_FORM_BUTTON(__dirname + '/account_circle'),
        opacity: 1,
    }

});
