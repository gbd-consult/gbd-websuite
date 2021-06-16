module.exports = v => ({
    '.cmpForm': {

        borderCollapse: 'collapse',
        width: '100%',

        'tr.isError': {
            '.uiControlBox': {
                borderColor: v.ERROR_COLOR,
            }
        },

        'td, th': {
            verticalAlign: 'middle',
            padding: [v.UNIT2, 0, v.UNIT2, 0],
            //fontSize: v.SMALL_FONT_SIZE,
            //borderWidth: 1,
            borderStyle: 'dotted',
            borderColor: v.BORDER_COLOR,
            textAlign: 'left',
            lineHeight: '120%',
        },

        'td': {
            maxWidth: 300,

        },
        'th': {
            fontWeight: 'bold',
            paddingRight: v.UNIT2,
            maxWidth: 100,

        },

        'tr.cmpFormError': {
            'td, th': {
                fontSize: v.TINY_FONT_SIZE,
                color: v.ERROR_COLOR,
                opacity: 0,
                padding: 0,
                ...v.TRANSITION('opacity'),
            },

            '&.isActive': {
                'td, th': {
                    opacity: 1,
                    padding: [v.UNIT, 0, v.UNIT, 0],
                ...v.TRANSITION('opacity'),
                }
            }

        },

    }


});