module.exports = v => {

    let norm = v.COLOR.blueGrey500;
    let active = v.COLOR.blue300;

    let baseColor = v.COLOR.blueGrey500;

    let common = {
        stroke: baseColor,
        strokeWidth: 2,
    };

    let mark = {
        mark: 'circle',
        markFill: baseColor,
        markSize: 10,
    };

    return {
        '.modLensFeature': {
            ...common,
            ...mark,
            fill: v.COLOR.opacity(baseColor, 0.3),
        },


        '.modLensAnchor': {
            ...v.ICON('normal'),
            ...v.LOCAL_SVG('move', v.COLOR.white),
            borderRadius: v.BORDER_RADIUS,
            border: '3px solid white',
            backgroundColor: v.COLOR.opacity(baseColor, 0.5),
        },

        '.modLensButtonPoint': {
            ...v.LOCAL_SVG('g_point', norm),
            '&.isActive': {
                ...v.LOCAL_SVG('g_point', active),
            },
        },

        '.modLensButtonLineString': {
            ...v.LOCAL_SVG('g_line', norm),
            '&.isActive': {
                ...v.LOCAL_SVG('g_line', active),
            },
        },

        '.modLensButtonBox': {
            ...v.LOCAL_SVG('g_box', norm),
            '&.isActive': {
                ...v.LOCAL_SVG('g_box', active),
            },
        },

        '.modLensButtonPolygon': {
            ...v.LOCAL_SVG('g_poly', norm),
            '&.isActive': {
                ...v.LOCAL_SVG('g_poly', active),
            },
        },

        '.modLensButtonCircle': {
            ...v.LOCAL_SVG('g_circle', norm),
            '&.isActive': {
                ...v.LOCAL_SVG('g_circle', active),
            },
        },


        '.modLensCancelButton': {
            ...v.CLOSE_SVG(v.COLOR.blueGrey500),
        },

    }
};
