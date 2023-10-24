module.exports = v => {

    let FIXED = {
        position: 'sticky',
        backgroundColor: v.COLOR.blueGrey100,
        '.uiText': {
            fontWeight: 600,
        },
    }

    return {
        '.uiTable': {
            width: '100%',
            height: '100%',
            overflow: 'auto',

            '> table': {
                borderCollapse: 'collapse',

                '> thead, > tfoot': {
                    ...FIXED,
                    'tr:nth-child(2)': {
                        backgroundColor: v.COLOR.blueGrey50,
                    },
                    'td.uiTableCell': {
                        borderTopWidth: 0,
                        borderBottomWidth: 0,
                    },
                    zIndex: 2,
                },

                '> thead': {
                    top: 0,
                },

                '> tfoot': {
                    bottom: 0,
                },

                'td.uiTableCell': {
                    verticalAlign: 'top',
                    border: [1, 'solid', v.COLOR.blueGrey50],
                    minWidth: v.UNIT * 10,

                    '.uiControlBox': {
                        borderWidth: 0,
                    },
                    '.uiRawInput': {
                        minWidth: v.UNIT * 20,
                    },
                    '.uiText': {
                        display: 'flex',
                        flexDirection: 'row',
                        alignItems: 'center',
                        height: v.CONTROL_SIZE,
                        padding: v.UNIT2,
                        width: '100%',
                    }
                },

                'tbody > tr > td.uiTableCell': {
                    '.uiText': {
                        color: v.LABEL_COLOR,
                    }
                },
                'txd.uiTableCell:last-child': {
                    width: '100%'
                },
            }
        },

        '.uiTable.uiTableWithFixedLeftColumn td.uiTableCell:first-child': {
            ...FIXED,
            left: 0,
            zIndex: 1,
        },
        '.uiTable.uiTableWithFixedRightColumn td.uiTableCell:last-child': {
            ...FIXED,
            right: 0,
            zIndex: 1,
        },
    }
}
