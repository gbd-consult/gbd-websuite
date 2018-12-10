module.exports = v => {

    let vectorDemo = {
        stroke: v.COLOR.cyan300,
        strokeWidth: 1,
        fill: v.COLOR.opacity(v.COLOR.cyan50, 0.3),
        labelFill: v.COLOR.cyan500,
        labelFontSize: 12,
        labelBackground: v.COLOR.cyan100,
        labelPadding: 2,
        //labelMaxResolution: 0.1,
    };


    return {

        '.modInfobar': {
            background: v.COLOR.grey800
        },

        '.vectorDemo': vectorDemo,

        '.vectorDemoEdit': {
            ...vectorDemo,
            mark: 'circle',
            markSize: 8,
            markFill: v.COLOR.black,
        },

        /*
            // test for framed gws
            // @TODO: mobile selectors should look the the gws width, not the media
            '': {
                width: 800,
                height: 800,
                margin: 50,
                border: '1px solid blue',

            },
         */


    }
};