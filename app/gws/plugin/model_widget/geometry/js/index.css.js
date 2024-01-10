module.exports = v => ({

    '.cmpFormDrawGeometryButton': {
        ...v.ROUND_FORM_BUTTON(__dirname + '/draw_black_24dp'),
    },

    '.cmpFormEditGeometryButton': {
        ...v.ROUND_FORM_BUTTON(v.ZOOM_ICON),
    },

    '.cmpFormGeometryTextButton': {
        ...v.ROUND_FORM_BUTTON(__dirname + '/fact_check_FILL0_wght400_GRAD0_opsz24'),
    },
});
