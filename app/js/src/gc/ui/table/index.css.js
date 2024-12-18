module.exports = v => {

    let FIXED = {
        position: 'sticky',
        backgroundColor: v.COLOR.blueGrey100,
        '.uiTableCell': {
            fontWeight: 600,
        },
    }

    return {
        '.uiTable': {
            width: '100%',
            height: '100%',
            overflow: 'auto',
            borderBottom: [1, 'solid', v.COLOR.blueGrey50],


            '> table': {
                borderCollapse: 'collapse',
                minWidth: '100%',

                '> thead, > tfoot': {
                    ...FIXED,
                    'tr:nth-child(2)': {
                        backgroundColor: v.COLOR.blueGrey50,
                    },
                    'td.uiTableTd': {
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

                'td.uiTableTd': {
                    verticalAlign: 'top',
                    border: [1, 'solid', v.COLOR.blueGrey50],

                    '.uiRawInput': {
                        minWidth: v.UNIT * 20,
                    },

                    '.uiTableCell': {
                        display: 'flex',
                        flexDirection: 'row',
                        alignItems: 'center',
                        minHeight: v.CONTROL_SIZE,
                        padding: v.UNIT2,
                        width: '100%',

                        '&.uiAlignCenter': {
                            justifyContent: 'center',
                        },
                        '&.uiAlignRight': {
                            justifyContent: 'rig    ht',
                        },
                    }
                },

                'thead > tr > td.uiTableTd': {
                    '.uiControlBox': {
                        borderWidth: 0,
                    },
                },

                'tbody > tr:nth-child(even)': {
                    backgroundColor: v.EVEN_STRIPE_COLOR,

                },

                'tbody > tr.isSelected > td.uiTableTd': {
                    backgroundColor: v.EVEN_STRIPE_COLOR,
                    '> .uiControl': {
                        backgroundColor: 'white',
                    },

                },
            }
        },

        '.uiTable.withSelection > table > tbody > tr': {
            opacity: 0.7,
        },
        '.uiTable.withSelection > table > tbody > tr.isSelected': {
            opacity: 1,
        },
        '.uiTable.withFixedLeftColumn td.uiTableTd:first-child': {
            ...FIXED,
            left: 0,
            zIndex: 1,
        },
        '.uiTable.withFixedRightColumn td.uiTableTd:last-child': {
            ...FIXED,
            right: 0,
            zIndex: 1,
        },
        '.uiTable tbody > tr > td.uiTableTd': {
            '> .uiControl': {
                margin: v.UNIT2,

            },
            '> .cmpFormList': {
                margin: v.UNIT2,

            }
        },
    }
}
