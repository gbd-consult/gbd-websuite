module.exports = v => ({

    '.cmpDescription': {
        'p, h1, h2, h3': {
            margin: [0, 0, v.UNIT4, 0],
            lineHeight: '120%',
            overflowWrap: 'break-word',
            wordWrap: 'break-word',
            fontWeight: 'normal'
        },

        'li': {
            lineHeight: '120%',
            margin: [v.UNIT, 0, v.UNIT, v.UNIT * 8],
        },

        'h1': {
            fontSize: v.BIG_FONT_SIZE,
        },

        'h2': {
            fontWeight: '800'
        },

        'p.head': {
            fontSize: v.BIG_FONT_SIZE,
        },

        'p.head2': {
            fontWeight: '800'
        },

        'p.text2': {
            fontSize: v.SMALL_FONT_SIZE,
        },

        'img': {
            maxWidth: '100%',
        },

        'table': {
            margin: [0, 0, v.UNIT4, 0],
            borderCollapse: 'collapse',
            width: '100%',
        },

        'table tr.thead td': {
            backgroundColor: v.BORDER_COLOR,
        },

        'td, th': {
            verticalAlign: 'top',
            fontSize: v.SMALL_FONT_SIZE,
            padding: v.UNIT2,
            borderWidth: 1,
            borderStyle: 'dotted',
            borderColor: v.BORDER_COLOR,
            maxWidth: 300,
            overflow: 'hidden',
            textAlign: 'left',
            lineHeight: '120%',
            overflowWrap: 'break-word',
        },

        'th': {
            fontWeight: 'bold',
        },

        '.metaurl': {
            fontSize: v.SMALL_FONT_SIZE,
            textAlign: 'right',

            'a': {
                textDecoration: 'none',

                '&::before': {
                    content: "'metadata ▹'"
                }
            }
        },

        '.subsection': {
            paddingTop: v.UNIT4,
            paddingBottom: v.UNIT4,
            borderTopWidth: 1,
            borderTopColor: v.BORDER_COLOR,
            borderTopStyle: 'dotted',
            fontSize: v.SMALL_FONT_SIZE,

            '.head': {
                fontWeight: '800',
                fontSize: v.NORMAL_FONT_SIZE,
            }
        },

    },

});

