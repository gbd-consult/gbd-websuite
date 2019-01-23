import * as React from 'react';

import * as gws from 'gws';

const SCALE_RULER_MAX_WIDTH = 200;
const ATTRIBUTIONS_SEPARATOR = ' | ';

interface ScaleRulerViewProps extends gws.types.ViewProps {
    mapResolution: number,
}

class ScaleRulerView extends gws.View<ScaleRulerViewProps> {

    round(m) {
        let s = Math.pow(10, Math.floor(Math.log(m) * Math.LOG10E)),
            t = Math.floor(m / s);
        if (t > 5)
            t = 5;
        else if (t > 2)
            t = 2;
        return t * s;
    }

    render() {
        let res = this.props.mapResolution;

        if (!res)
            return null;

        let w = SCALE_RULER_MAX_WIDTH,
            m = this.round(res * w),
            width = Math.round(m / res),
            label = (m >= 1000) ? Math.floor(m / 1000) + 'km' : m + 'm';

        return (
            <div className="modDecorationScaleRuler">
                <div className="modDecorationScaleRulerLabel">{label}</div>
                <div className="modDecorationScaleRulerBar" style={{width}}/>
            </div>
        );
    }
}

class ScaleRuler extends gws.Controller {
    get defaultView() {
        return this.createElement(
            this.connect(ScaleRulerView,
                ['mapResolution']));
    }
}

interface AttributionViewProps extends gws.types.ViewProps {
    mapAttribution: Array<string>,
}

class AttributionView extends gws.View<AttributionViewProps> {

    render() {
        let a = this.props.mapAttribution;

        if (!a || !a.length)
            return null;

        return (
            <div className="modDecorationAttribution">
                <gws.ui.HtmlBlock content={a.join(ATTRIBUTIONS_SEPARATOR)} /></div>
        );
    }
}

class Attribution extends gws.Controller {
    get defaultView() {
        return this.createElement(
            this.connect(AttributionView,
                ['mapAttribution']));
    }
}

export const tags = {
    'Decoration.ScaleRuler': ScaleRuler,
    'Decoration.Attribution': Attribution,
};

