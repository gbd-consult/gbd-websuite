module.exports = v => ({
    '.cmpFileList': {
        width: '100%',
        '.cmpFile': {
            margin: v.UNIT,
        }
    },

    '.cmpFileListRow': {
        overflowX: 'auto',
        '.cmpFileListInner': {
            display: 'flex',
        },
    },

    '.cmpFileListGrid': {
        '.cmpFile': {
            float: 'left',
        },
    },

    '.cmpFile': {

        backgroundColor: v.EVEN_STRIPE_COLOR,

        textAlign: 'center',
        ...v.TRANSITION(),

        ':hover': {
            backgroundColor: v.COLOR.blueGrey100,
        },

        '&.isSelected, &.isSelected:hover': {
            backgroundColor: v.SELECTED_ITEM_BACKGROUND,
            '.cmpFileLabel': {
                backgroundColor: v.SELECTED_ITEM_BACKGROUND,
                color: v.COLOR.blue600,

            }
        },
    },


    '.cmpFileContent': {
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

    '.cmpFileLabel': {
        width: v.UNIT * 30,
        padding: v.UNIT2,
        fontSize: v.TINY_FONT_SIZE,
        // backgroundColor: v.BORDER_COLOR,
        // color: 'white',
        overflow: 'hidden',


    },


    '.cmpFile_any .cmpFileContent': { ...v.SVG(__dirname + '/file_any', v.ICON_COLOR) },
    '.cmpFile_pdf .cmpFileContent': { ...v.SVG(__dirname + '/file_pdf', v.ICON_COLOR) },
    '.cmpFile_csv .cmpFileContent': { ...v.SVG(__dirname + '/file_csv', v.ICON_COLOR) },
    '.cmpFile_txt .cmpFileContent': { ...v.SVG(__dirname + '/file_txt', v.ICON_COLOR) },
    '.cmpFile_zip .cmpFileContent': { ...v.SVG(__dirname + '/file_zip', v.ICON_COLOR) },
    '.cmpFile_doc .cmpFileContent': { ...v.SVG(__dirname + '/file_doc', v.ICON_COLOR) },
    '.cmpFile_xls .cmpFileContent': { ...v.SVG(__dirname + '/file_xls', v.ICON_COLOR) },
    '.cmpFile_ppt .cmpFileContent': { ...v.SVG(__dirname + '/file_ppt', v.ICON_COLOR) },

});