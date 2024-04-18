import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

interface SmallMapProps extends gws.types.ViewProps {
    boxSize?: Array<number>;
}

interface SmallMapState {
    inited: boolean;
}

const POINTER_DEBOUNCE = 800;
const MIN_MAP_HEIGHT = 100;
const MAX_MAP_HEIGHT = 500;
const MIN_BOX_SIZE = 10;


const OSM_SCALES = [
    150000000,
    70000000,
    35000000,
    15000000,
    10000000,
    4000000,
    2000000,
    1000000,
    500000,
    250000,
    150000,
    70000,
    35000,
    15000,
    8000,
    4000,
    2000,
    1000,
    500,
]


export class SmallMap extends React.Component<SmallMapProps, SmallMapState> {

    smallMap: gws.types.IMapManager | null = null;
    mainMap: gws.types.IMapManager;
    mapRef: React.RefObject<HTMLDivElement>;
    oMap: ol.Map = null;
    oProjection: ol.proj.Projection;
    oOverlay: ol.Overlay;
    state: SmallMapState = {inited: false}

    get app() {
        return this.props.controller.app;
    }

    __(key) {
        return this.props.controller.__(key);
    }

    constructor(props) {
        super(props);
        this.mapRef = React.createRef();
    }

    async componentDidMount() {
        if (!this.state.inited)
            await this.init()

        if (!this.smallMap)
            return;

        if (this.mapRef.current) {
            this.setMapTarget(this.mapRef.current);
        }

        this.refresh()
    }

    componentDidUpdate() {
        if (!this.state.inited)
            return;

        this.refresh()
    }

    async init() {
        if (!this.app.project.overviewMap)
            return;

        // if (this.props.boxSize) {
        //     this.app.project.overviewMap.resolutions = OSM_SCALES.map(s => gws.lib.scale2res(s))
        //     this.app.project.overviewMap.rootLayer.resolutions = OSM_SCALES.map(s => gws.lib.scale2res(s))
        // }

        this.mainMap = this.props.controller.map;

        this.smallMap = new gws.MapManager(this.app, false);
        await this.smallMap.init(this.app.project.overviewMap, {});

        this.smallMap.oMap.addInteraction(new ol.interaction.Pointer({
            handleDownEvent: e => this.handleMouseEvent(e),
            handleDragEvent: e => this.handleMouseEvent(e),

        }));

        this.oProjection = this.smallMap.oMap.getView().getProjection();

        let d = document.createElement('div');
        d.className = 'overviewBox';

        let c = document.createElement('div');
        c.className = 'overviewBoxCenter';

        d.appendChild(c);


        this.oOverlay = new ol.Overlay({
            element: d,
            stopEvent: false,
            positioning: 'top-left',
        });

        this.smallMap.oMap.addOverlay(this.oOverlay);

        this.app.whenChanged('mapUpdateCount', () => this.refresh());
        this.app.whenChanged('windowSize', () => this.refresh());
        this.setState({inited: true})

    }

    setMapTarget(div) {

        if (this.props.boxSize) {
            div.style.height = div.offsetWidth + 'px';
        } else {
            let e = this.app.project.overviewMap.extent;
            let f = ol.extent.getWidth(e) / ol.extent.getHeight(e);
            let h = gws.lib.clamp(div.offsetWidth / f, MIN_MAP_HEIGHT, MAX_MAP_HEIGHT);
            div.style.height = (h | 0) + 'px';
        }

        this.smallMap.oMap.setTarget(div);
    }


    render() {
        return this.smallMap && <div className="overviewMap" ref={this.mapRef}/>
    }


    mouseTimer: any = 0;

    handleMouseEvent(evt: ol.MapBrowserPointerEvent) {
        clearTimeout(this.mouseTimer);

        // prevent excessive map updates when dragging

        let cc = ol.proj.transform(
            evt.coordinate,
            this.oProjection,
            this.mainMap.projection,
        );

        this.oOverlay.setPosition(cc);

        this.mouseTimer = setTimeout(
            () => this.mainMap.setViewState({center: cc}),
            POINTER_DEBOUNCE);

        return true;
    }

    refresh() {
        clearTimeout(this.mouseTimer);

        let vs = this.mainMap.viewState;

        this.smallMap.setRotation(vs.rotation);

        let w, h;

        if (this.props.boxSize) {
            let div = this.mapRef.current;

            if (!div)
                return;

            let [mw, mh] = this.props.boxSize;

            if (mw > mh) {
                w = div.offsetWidth * 0.9
                h = w * (mh / mw)
            } else {
                h = div.offsetWidth * 0.9
                w = h * (mw / mh)
            }

            this.smallMap.setResolution(vs.resolution * (mw / w))
            this.smallMap.setCenter([vs.centerX, vs.centerY])

        } else {

            let size = vs.size
            let res = vs.resolution / this.smallMap.oMap.getView().getResolution();
            w = Math.max(MIN_BOX_SIZE, (size[0] * res) | 0);
            h = Math.max(MIN_BOX_SIZE, (size[1] * res) | 0);

        }

        let el = this.oOverlay.getElement() as HTMLDivElement;
        el.style.width = w + 'px';
        el.style.height = h + 'px';
        el.style.left = (-w / 2) + 'px';
        el.style.top = (-h / 2) + 'px';

        let cc = ol.proj.transform(
            [vs.centerX, vs.centerY],
            this.mainMap.projection,
            this.oProjection,
        );

        this.oOverlay.setPosition(cc);
    }
}
