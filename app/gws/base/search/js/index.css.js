module.exports = v => ({
    '.searchSidebarIcon': {
        ...v.SIDEBAR_ICON('google:action/search')
    },

    '.searchClearButton.uiIconButton': {
        ...v.ICON_SIZE('small'),
        ...v.SVG(v.CLOSE_ICON, v.BORDER_COLOR),
    },

    '.searchWaitButton.uiIconButton': {
        ...v.ICON_SIZE('small'),
        // backgroundImage: v.IMAGE('ajax.gif'),
    },

    '.searchBox': {
        '.uiInput .uiControlBox': {
            borderWidth: 0
        },

    },


















    '.searchOptions': {
        display: 'none',
        // overflow: 'hidden',
        // ...v.TRANSITION(),
    },
    '.searchBox.withOptions .searchOptions': {
        backgroundColor: v.EVEN_STRIPE_COLOR,
        display: 'block',
    },





    '.searchIcon.uiIconButton': {
        ...v.ICON_SIZE('small'),
        ...v.SVG(v.SEARCH_ICON, v.BORDER_COLOR),
        ...v.TRANSITION(),

        '.hasOptions&': {
            ...v.SVG('google:navigation/chevron_right', v.BORDER_COLOR),
            transform: 'rotate(90deg)',
        },

        '.withOptions&': {
            transform: 'rotate(-90deg)',
        }

    },

    '.searchAltbarDropDown': {
        maxHeight: 300,
        overflow: 'auto',
        overflowX: 'hidden',
        backgroundColor: v.COLOR.white,
        borderTop: [1, 'solid', v.BORDER_COLOR],
    },



    '.searchSideButton': {
        minWidth: v.CONTROL_SIZE,
        maxWidth: v.CONTROL_SIZE,
    },


    '.searchAltbar': {
        backgroundColor: v.COLOR.white,
        border: [1, 'solid', v.BORDER_COLOR],
    },



    '.searchResultsEmpty': {
        textAlign: 'center',
        color: v.BORDER_COLOR,
        fontStyle: 'italic',
        padding: v.UNIT4,

    },

    '.searchResults .cmpListContent': {
        padding: [v.UNIT4, v.UNIT4, v.UNIT4, 0],
    },

    '.searchSidebar': {
        '.modSidebarTabHeader': {
            padding: [v.UNIT, v.UNIT2, v.UNIT, v.UNIT2],
        }
    },

    '.searchResultsTeaser': {
        position: 'relative',
        'p.head': {
            color: v.FOCUS_COLOR,
            cursor: 'pointer',
            fontSize: v.NORMAL_FONT_SIZE,
            marginBottom: v.UNIT,
        },
        'p': {
            fontSize: v.SMALL_FONT_SIZE,
            lineHeight: '120%'
        },
    },

});
