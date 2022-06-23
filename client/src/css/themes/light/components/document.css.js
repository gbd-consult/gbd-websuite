module.exports = v => ({
    '.cmpDocumentList': {
        width: '100%',
        '.cmpDocument': {
            margin: v.UNIT,
        }
    },

    '.cmpDocumentListRow': {
        overflowX: 'auto',
        '.cmpDocumentListInner': {
            display: 'flex',
        },
    },

    '.cmpDocumentListGrid': {
        '.cmpDocument': {
            float: 'left',
        },
    },

    '.cmpDocument': {

        backgroundColor: v.EVEN_STRIPE_COLOR,
        backgroundSize: [v.UNIT * 10, v.UNIT * 10],
        backgroundRepeat: 'no-repeat',
        backgroundPosition: 'center center',

        textAlign: 'center',
        ...v.TRANSITION(),

        ':hover': {
            backgroundColor: v.HOVER_COLOR,
        },

        '&.isSelected, &.isSelected:hover': {
            backgroundColor: v.BORDER_COLOR,
            '.cmpDocumentLabel': {
                color: 'white',
            }
        },
    },


    '.cmpDocumentContent': {
        padding: v.UNIT2,
        textAlign: 'center',
        width: v.UNIT * 30,
        height: v.UNIT * 30,
        display: 'inline-block',

        'img': {
            width: '100%',
            height: '100%',
            objectFit: 'cover',
        }
    },

    '.cmpDocumentLabel': {
        width: '100%',
        padding: v.UNIT2,
        fontSize: v.TINY_FONT_SIZE,

    },


    '.cmpDocument_any': { ...v.SVG('document_any', v.COLOR.red) },
    '.cmpDocument_pdf': { ...v.SVG('document_pdf', v.ICON_COLOR) },
    '.cmpDocument_csv': { ...v.SVG('document_csv', v.ICON_COLOR) },
    '.cmpDocument_txt': { ...v.SVG('document_txt', v.ICON_COLOR) },
    '.cmpDocument_zip': { ...v.SVG('document_zip', v.ICON_COLOR) },
    '.cmpDocument_doc': { ...v.SVG('document_doc', v.ICON_COLOR) },
    '.cmpDocument_xls': { ...v.SVG('document_xls', v.ICON_COLOR) },
    '.cmpDocument_ppt': { ...v.SVG('document_ppt', v.ICON_COLOR) },


});