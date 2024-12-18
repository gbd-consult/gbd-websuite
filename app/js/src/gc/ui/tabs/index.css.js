module.exports = v => ({

    '.uiTabs': {
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
    },

    '.uiTabs.hasVBox .uiTabContent': {
        overflow: 'hidden'
    },

    '.uiTabsHead': {
        display: 'flex',
        alignItems: 'center',
        width: '100%',
        marginBottom: v.UNIT8,
    },

    '.uiTabHeadItem': {
        textTransform: 'uppercase',
        borderBottom: 3,
        borderBottomStyle: 'solid',
        borderBottomColor: v.COLOR.opacity(v.FOCUS_COLOR, 0),
        ...v.TRANSITION(),

        '.uiRawButton': {
            padding: [v.UNIT4, v.UNIT4, v.UNIT4, v.UNIT4],
        },

        '&:hover': {
            backgroundColor: v.HOVER_COLOR,
        },

        '&.isActive': {
            color: v.FOCUS_COLOR,
            borderBottomColor: v.FOCUS_COLOR,
        },

        '&.isDisabled': {
            opacity: 0.5,
        },
    },

    '.uiTabContent': {
        overflow: 'auto',
        flex: 1,
        // paddingRight: v.UNIT * 4,
    }

});
