module.exports = v => ({
    '.modIdentifyClickToolbarButton': {
        ...v.TOOLBAR_BUTTON(__dirname + '/identify')
    },

    '.modIdentifyHoverToolbarButton': {
        ...v.TOOLBAR_BUTTON(__dirname + '/identifyhover')
    },

    '.modIdentifyClickToolboxIcon': {
        ...v.TOOLBOX_ICON('google:action/info')
    },

    '.modIdentifyHoverToolboxIcon': {
        ...v.TOOLBOX_ICON('google:hardware/mouse')
    },
});
