module.exports = v => ({

    '.uiGrid': {
        '> .uiRow > .uiCell': {
            minHeight: v.UNIT * 10,
        },

        '.uiControlBox': {
            borderWidth: 0,
        },

        '.uiLabel': {
            display: 'none',
        },
    },


    '.uiTable': {
        width: '100%',
        height: '100%',
        overflow: 'hidden',

        '.uiGrid': {
            position: 'relative',
        }
    },

    '.uiTableCell': {
        border: [1, 'solid', v.BORDER_COLOR],
    },

    '.uiTableHead': {
        '.uiTableCell': {
            backgroundColor: v.COLOR.blueGrey100,
        },
    },

    '.uiTableFixed .uiTableCell': {
        backgroundColor: v.COLOR.blueGrey50,
    },

    '.uiTableStaticText': {
        display: 'flex',
        height: v.UNIT * 10,
        alignItems: 'center',
        paddingLeft: v.UNIT * 2,
        paddingRight: v.UNIT * 2,
    }

});