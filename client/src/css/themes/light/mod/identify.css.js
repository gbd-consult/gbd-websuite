module.exports = v => ({
    '.modIdentifyClickToolbarButton': {
        ...v.TOOLBAR_BUTTON('google:action/info')
    },

    '.modIdentifyHoverToolbarButton': {
        ...v.TOOLBAR_BUTTON('google:hardware/mouse')
    },

    '.modIdentifyClickToolboxIcon': {
        ...v.TOOLBOX_ICON('google:action/info')
    },

    '.modIdentifyHoverToolboxIcon': {
        ...v.TOOLBOX_ICON('google:hardware/mouse')
    },
});
