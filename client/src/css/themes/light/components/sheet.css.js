module.exports = v => ({
    '.cmpPropertySheet': {

        borderCollapse: 'collapse',
        width: '100%',

        '.cmpSheetHead': {
            borderBottom: '1px solid ' + v.BORDER_COLOR,
            fontWeight: 800,

        },

        'td, th': {
            verticalAlign: 'middle',
            paddingTop: v.UNIT2,
            paddingBottom: v.UNIT2,
            //fontSize: v.SMALL_FONT_SIZE,
            //borderWidth: 1,
            borderStyle: 'dotted',
            borderColor: v.BORDER_COLOR,
            maxWidth: 300,
            textAlign: 'left',
            lineHeight: '120%',
        },

        'th': {
            fontWeight: 'bold',
            paddingRight: v.UNIT2,

        },

    }


});