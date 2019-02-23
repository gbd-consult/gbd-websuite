module.exports = v => {

    let button = icon => ({
        ...v.SVG(icon, v.DRAWBOX_BUTTON_COLOR),
        '&.isActive': {
            ...v.SVG(icon, v.DRAWBOX_ACTIVE_BUTTON_COLOR),
        },
    })

    return {

        '.modDimensionToolbarButton': {
            ...v.TOOLBAR_BUTTON('google:image/straighten')
        },

        '.modDimensionSidebarIcon': {
            ...v.SIDEBAR_ICON('google:image/straighten')
        },

        '.modDimensionFeature': {
            mark: 'circle',
            markFill: v.COLOR.opacity(v.COLOR.pink100, 0.3),
            markSize: 30,
        },
        '.modDimensionSelectedFeature': {


            mark: 'circle',
            markFill: v.COLOR.pink700,
            markSize: 20,
            markStroke: v.COLOR.pink100,
            markStrokeWidth: 5,


        },

        '.modDimensionModifyButton': button('cursor'),
        '.modDimensionLineButton': button('dim_line'),
        '.modDimensionArcButton': button('dim_arc'),
        '.modDimensionCircleButton': button('dim_circle'),
        '.modDimensionRemoveButton': button('google:action/delete'),


        '.modDimensionDimLabel': {
            fontSize: 12,
            fill: v.COLOR.lightBlue800,
            userSelect: 'none',
        },

        '.modDimensionDimLine': {
            strokeWidth: 2,
            stroke: v.COLOR.lightBlue800,
            fill: 'transparent',
        },

        '.modDimensionDimGuide': {
            strokeDasharray: '3,3',
            strokeWidth: 1,
            stroke: v.COLOR.lightBlue800,
            fill: 'transparent',
        },

        '.modDimensionDimMark': {
            strokeWidth: 4,
            stroke: v.COLOR.lightBlue700,
            fill: 'transparent',
        },


        /*
                        }
                        .dim {
                            stroke: #006FB8;
                            stroke-width: 2;
                        }
                        .add {
                            stroke: #006FB8;
                            stroke-width: 1;
                        }
                    </style>






         */

    }
}
