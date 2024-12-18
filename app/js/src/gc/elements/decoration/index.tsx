import * as React from 'react';

import * as gc from 'gc';

const SCALE_RULER_MAX_WIDTH = 200;
const ATTRIBUTIONS_SEPARATOR = ' | ';

interface ScaleRulerViewProps extends gc.types.ViewProps {
    mapResolution: number,
}

class ScaleRulerView extends gc.View<ScaleRulerViewProps> {

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

class ScaleRuler extends gc.Controller {
    get defaultView() {
        return this.createElement(
            this.connect(ScaleRulerView,
                ['mapResolution']));
    }
}

interface AttributionViewProps extends gc.types.ViewProps {
    mapAttribution: Array<string>,
}

class AttributionView extends gc.View<AttributionViewProps> {

    render() {
        let a = this.props.mapAttribution;

        if (!a || !a.length)
            return null;

        return (
            <div className="modDecorationAttribution">
                <gc.ui.HtmlBlock content={a.join(ATTRIBUTIONS_SEPARATOR)} /></div>
        );
    }
}

class Attribution extends gc.Controller {
    get defaultView() {
        return this.createElement(
            this.connect(AttributionView,
                ['mapAttribution']));
    }
}

class DecorationController extends gc.Controller {
    get defaultView() {
        return this.renderChildren();
    }



}

gc.registerTags({
    'Decoration': DecorationController,
    'Decoration.ScaleRuler': ScaleRuler,
    'Decoration.Attribution': Attribution,
});

