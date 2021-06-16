module.exports = v => ({
    '.uiVBox': {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        height: '100%',
    },

    '.uiVRow': {
        width: '100%',
        overflowY: 'auto',
        marginBottom: v.UNIT4,
        paddingRight: v.UNIT2,
        '> .uiCell': {
            marginLeft: v.UNIT4,
            '&:first-child': {
                marginLeft: 0,
            },
        }
    },


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

    '.uiCell.isSpaced': {
        marginLeft: v.UNIT4,
    },

    '.uiForm': {
        '>.uiRow': {
            margin: [0, 0, v.UNIT4, 0],
            '&:last-child': {
                margin: [0, 0, 0, 0],
            },

            '>.uiCell': {
                marginLeft: v.UNIT4,
                '&:first-child': {
                    marginLeft: 0,
                },
            }
        }
    },

    '.uiForm.isTabular': {
        display: 'table',
        width: '100%',

        '.uiTabularSpacer': {
            display: 'table-row',
            'div': {
                display: 'table-cell',
                padding: v.UNIT2,
            }
        },

        '> .uiControl': {
            display: 'table-row',

            '> .uiControlBody': {
                width: '100%',
                verticalAlign: 'middle',
                display: 'table-cell',
            },

            '> .uiLabel': {
                verticalAlign: 'middle',
                display: 'table-cell',
                padding: [0, v.UNIT4, 0, 0],
            },
        },
    },

    '.uiGroup': {
        ' > .uiControlBody > .uiControlBox': {
            height: 'auto',
            padding: [0, v.UNIT2, 0, 0],
        },

        '&.noBorder > .uiControlBody >.uiControlBox': {
            border: 'none',
        },

        '&.isVertical > .uiControlBody > .uiControlBox': {
            flexDirection: 'column',
            alignItems: 'normal',
            height: 'auto',
            padding: [v.UNIT2, v.UNIT2, v.UNIT2, 0],

        },
    }


});