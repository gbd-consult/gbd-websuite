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

        textAlign: 'center',
        ...v.TRANSITION(),

        ':hover': {
            backgroundColor: v.COLOR.blueGrey100,
        },

        '&.isSelected, &.isSelected:hover': {
            backgroundColor: v.SELECTED_ITEM_BACKGROUND,
            '.cmpDocumentLabel': {
                backgroundColor: v.SELECTED_ITEM_BACKGROUND,
                color: v.COLOR.blue600,

            }
        },
    },


    '.cmpDocumentContent': {
        padding: v.UNIT2,
        textAlign: 'center',
        width: v.UNIT * 30,
        height: v.UNIT * 30,

        backgroundRepeat: 'no-repeat',
        backgroundPosition: 'center center',
        backgroundSize: [v.UNIT * 10, v.UNIT * 10],


        'img': {
            width: '100%',
            height: '100%',
            objectFit: 'cover',
        }
    },

    '.cmpDocumentLabel': {
        width: v.UNIT * 30,
        padding: v.UNIT2,
        fontSize: v.TINY_FONT_SIZE,
        // backgroundColor: v.BORDER_COLOR,
        // color: 'white',
        overflow: 'hidden',


    },


    '.cmpDocument_any .cmpDocumentContent': { ...v.SVG('document_any', v.ICON_COLOR) },
    '.cmpDocument_pdf .cmpDocumentContent': { ...v.SVG('document_pdf', v.ICON_COLOR) },
    '.cmpDocument_csv .cmpDocumentContent': { ...v.SVG('document_csv', v.ICON_COLOR) },
    '.cmpDocument_txt .cmpDocumentContent': { ...v.SVG('document_txt', v.ICON_COLOR) },
    '.cmpDocument_zip .cmpDocumentContent': { ...v.SVG('document_zip', v.ICON_COLOR) },
    '.cmpDocument_doc .cmpDocumentContent': { ...v.SVG('document_doc', v.ICON_COLOR) },
    '.cmpDocument_xls .cmpDocumentContent': { ...v.SVG('document_xls', v.ICON_COLOR) },
    '.cmpDocument_ppt .cmpDocumentContent': { ...v.SVG('document_ppt', v.ICON_COLOR) },


});