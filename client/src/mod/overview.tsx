import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from './sidebar';
import * as types from "gws/types";

let {Form, Row, Cell} = gws.ui.Layout;

interface ViewProps extends gws.types.ViewProps {
    controller: SidebarOverviewController;
    mapUpdateCount: number;
    mapEditScale: number;
    mapEditAngle: number;
    mapEditCenterX: number;
    mapEditCenterY: number;

}

const POINTER_DEBOUNCE = 800;
const MIN_MAP_HEIGHT = 100;
const MAX_MAP_HEIGHT = 500;
const MIN_BOX_SIZE = 10;


interface SmallMapProps extends gws.types.ViewProps {
    boxSize?: Array<number>;
}

interface SmallMapState {
    inited: boolean;
}

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

        if (this.props.boxSize) {
            this.app.project.overviewMap.resolutions = OSM_SCALES.map(s => gws.tools.scale2res(s))
            this.app.project.overviewMap.layers[0].resolutions = OSM_SCALES.map(s => gws.tools.scale2res(s))
        }

        this.mainMap = this.props.controller.map;

        this.smallMap = new gws.MapManager(this.app, false);
        await this.smallMap.init(this.app.project.overviewMap, {});

        this.smallMap.oMap.addInteraction(new ol.interaction.Pointer({
            handleDownEvent: e => this.handleMouseEvent(e),
            handleDragEvent: e => this.handleMouseEvent(e),

        }));

        this.oProjection = this.smallMap.oMap.getView().getProjection();

        let d = document.createElement('div');
        d.className = 'modOverviewBox';

        let c = document.createElement('div');
        c.className = 'modOverviewBoxCenter';

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
            let h = gws.tools.clamp(div.offsetWidth / f, MIN_MAP_HEIGHT, MAX_MAP_HEIGHT);
            div.style.height = (h | 0) + 'px';
        }

        this.smallMap.oMap.setTarget(div);
    }


    render() {
        return this.smallMap && <div className="modOverviewMap" ref={this.mapRef}/>
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


class SidebarBody extends gws.View<ViewProps> {


    submit() {
        let map = this.props.controller.map;

        map.setViewState({
            center: [this.props.mapEditCenterX, this.props.mapEditCenterY],
            scale: this.props.mapEditScale,
            angle: this.props.mapEditAngle
        }, true)
    }

    mapInfoBlock() {
        let map = this.props.controller.map;

        let coord = n => map.formatCoordinate(Number(n) || 0);
        let ext = map.viewExtent.map(coord).join(', ');

        let res = map.resolutions,
            maxScale = Math.max(...res.map(gws.tools.res2scale)),
            minScale = Math.min(...res.map(gws.tools.res2scale));

        let bind = k => ({
            whenChanged: v => this.props.controller.update({[k]: v}),
            whenEntered: v => this.submit(),
            value: this.props[k]
        });


        return <Form>
            <Row>
                <Cell flex>
                    <Form tabular>
                        <gws.ui.TextInput
                            label={this.__('modOverviewProjection')}
                            value={map.projection.getCode()}
                            readOnly/>
                        <gws.ui.TextInput
                            label={this.__('modOverviewExtent')}
                            value={ext}
                            readOnly/>
                        <gws.ui.TextInput
                            label={this.__('modOverviewCenterX')}
                            {...bind('mapEditCenterX')}/>
                        <gws.ui.TextInput
                            label={this.__('modOverviewCenterY')}
                            {...bind('mapEditCenterY')}/>
                        <gws.ui.NumberInput
                            minValue={minScale}
                            maxValue={maxScale}
                            step={1000}
                            label={this.__('modOverviewScale')}
                            {...bind('mapEditScale')}/>
                        <gws.ui.NumberInput
                            minValue={0}
                            maxValue={359}
                            step={5}
                            withClear
                            label={this.__('modOverviewRotation')}
                            {...bind('mapEditAngle')}/>
                    </Form>
                </Cell>


            </Row>
            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.Button
                        primary
                        whenTouched={() => this.submit()}
                        label={this.__('modOverviewUpdateButton')}
                    />
                </Cell>
            </Row>
        </Form>;
    }

    render() {
        let desc = this.props.controller.app.project.description;

        return <sidebar.Tab>

            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modOverviewSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <SmallMap controller={this.props.controller}/>
                {this.mapInfoBlock()}
            </sidebar.TabBody>

            {desc && <sidebar.TabFooter>
                <div className="modOverviewTabFooter">
                    <gws.ui.TextBlock className="cmpDescription" withHTML content={desc}/>
                </div>
            </sidebar.TabFooter>}


        </sidebar.Tab>
    }
}

class SidebarOverviewController extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'modOverviewSidebarIcon';

    async init() {
        this.app.whenChanged('mapUpdateCount', () => this.refresh());
        this.app.whenChanged('windowSize', () => this.refresh());
    }

    get tooltip() {
        return this.__('modOverviewSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarBody, [
                'mapUpdateCount',
                'mapEditScale',
                'mapEditAngle',
                'mapEditCenterX',
                'mapEditCenterY',
            ]),
        );
    }

    refresh() {
        let vs = this.map.viewState;

        this.update({
            'mapEditScale': vs.scale,
            'mapEditAngle': vs.angle,
            'mapEditCenterX': vs.centerX,
            'mapEditCenterY': vs.centerY,
        });
    }

}

export const tags = {
    'Sidebar.Overview': SidebarOverviewController
};

