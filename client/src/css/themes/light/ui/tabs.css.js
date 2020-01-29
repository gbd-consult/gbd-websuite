module.exports = v => ({

    '.uiTabsHead': {
        display: 'flex',
        alignItems: 'center',
        width: '100%',
        marginBottom: v.UNIT8,
    },

    '.uiTabHeader': {
        textTransform: 'uppercase',
        borderBottom: 3,
        borderBottomStyle: 'solid',
        borderBottomColor: v.COLOR.opacity(v.FOCUS_COLOR, 0),
        ...v.TRANSITION(),

        '.uiRawButton': {
            padding: [v.UNIT4, v.UNIT8, v.UNIT4, v.UNIT8],
        },

        '&:hover': {
            backgroundColor: v.HOVER_COLOR,
        },

        '&.isActive': {
            color: v.FOCUS_COLOR,
            borderBottomColor: v.COLOR.opacity(v.FOCUS_COLOR, 1),
        },

        '&.isDisabled': {
            opacity: 0.5,
        },
    },

    '.uiTabContent': {}

});
