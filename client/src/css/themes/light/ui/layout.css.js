module.exports = v => ({
    '.uiRow': {
        display: 'flex',
        alignItems: 'center',
        width: '100%',

    },

    '.uiDivider': {
        width: '100%',
        height: v.CONTROL_SIZE,
        display: 'flex',
        alignItems: 'center',
    },

    '.uiDividerInner': {
        width: '100%',
        height: 1,
        borderBottomWidth: 2,
        borderBottomStyle: 'dotted',
        borderBottomColor: v.BORDER_COLOR,
    },

    '.uiForm': {
        '.uiRow': {
            margin: [0, 0, v.UNIT4, 0],
            '&:last-child': {
                margin: [0, 0, 0, 0],
            }
        },

        '.uiCell': {
            marginLeft: v.UNIT4,
            '&:first-child': {
                marginLeft: 0,
            },
        }
    }
});