import * as React from 'react';
import * as ol from 'openlayers';

import * as gc from 'gc';
import * as toolbar from 'gc/elements/toolbar';
import * as components from 'gc/components';

let {Form, Row, Cell} = gc.ui.Layout;

const MASTER = 'Shared.MapShare';

const REFRESH_DELAY = 500;

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as Controller;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as Controller;
}

interface ViewProps extends gc.types.ViewProps {
    controller: Controller;
    mapshareDialogActive: boolean;
    mapshareTitle: string;
    mapshareUrl: string;
    mapshareQrCode: string;
    mapsharePending: boolean;
    mapshareCopied: boolean;
}

const StoreKeys = [
    'mapshareDialogActive',
    'mapshareTitle',
    'mapshareUrl',
    'mapshareQrCode',
    'mapsharePending',
    'mapshareCopied',
];

class MapShareTool extends gc.Tool {
    start() {
        this.map.prependInteractions([
            this.map.pointerInteraction({
                whenTouched: evt => _master(this).open(evt.coordinate),
            }),
        ]);
    }

    stop() {
    }
}

class MapShareDialog extends gc.View<ViewProps> {
    render() {
        if (!this.props.mapshareDialogActive)
            return null;

        let cc = _master(this);
        let ready = Boolean(this.props.mapshareUrl) && !this.props.mapsharePending;

        let close = () => cc.update({mapshareDialogActive: false});

        return <gc.ui.Dialog
            className="mapshareDialog"
            title={this.__('mapshareDialogTitle')}
            whenClosed={close}
        >
            <Form>
                <Row>
                    <Cell flex>
                        <gc.ui.TextInput
                            label={this.__('mapshareTitle')}
                            value={this.props.mapshareTitle}
                            whenChanged={v => cc.setTitle(v)}
                        />
                    </Cell>
                </Row>
                <Row>
                    <Cell flex>
                        <gc.ui.TextInput
                            label={this.__('mapshareUrl')}
                            value={this.props.mapshareUrl || ''}
                            readOnly
                        />
                    </Cell>
                </Row>
                <Row>
                    <Cell flex>
                        <div className="mapshareQrCodeBox">
                            {this.props.mapsharePending
                                ? <gc.ui.Loader/>
                                : this.props.mapshareQrCode && <img src={this.props.mapshareQrCode}/>}
                        </div>
                    </Cell>
                </Row>
                <Row>
                    <Cell flex/>
                    {1 && <Cell>
                        <gc.ui.Button
                            primary
                            disabled={!ready}
                            label={this.__('mapshareShareButton')}
                            whenTouched={() => cc.share()}
                        />
                    </Cell>}
                    <Cell>
                        <Cell flex/>
                        <gc.ui.Button
                            disabled={!ready}
                            label={this.__(this.props.mapshareCopied ? 'mapshareCopiedButton' : 'mapshareCopyButton')}
                            whenTouched={() => cc.copy()}
                        />
                    </Cell>
                </Row>
            </Form>
        </gc.ui.Dialog>;
    }
}

class Controller extends gc.Controller {
    uid = MASTER;
    shape: gc.gws.ShapeProps = null;
    scale = 0;
    requestCount = 0;
    refreshLater: Function;

    async init() {
        this.refreshLater = gc.lib.debounce(() => this.refresh(), REFRESH_DELAY);
        this.app.whenLoaded(() => this.whenAppLoaded());
    }

    async whenAppLoaded() {
        let setup = this.app.actionProps('mapshare') as gc.gws.plugin.mapshare.action.Props;

        if (!setup) {
            this.updateObject('toolbarHiddenItems', {'Toolbar.MapShare': true});
            return;
        }

        let link = this.app.urlParams[setup.linkParamName];
        if (link)
            await this.showLink(link);
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(MapShareDialog, StoreKeys));
    }

    open(coordinate) {
        this.shape = this.map.geom2shape(new ol.geom.Point(coordinate));
        this.scale = Math.round(this.map.viewState.scale);
        this.update({
            mapshareUrl: null,
            mapshareQrCode: null,
            mapshareCopied: false,
            mapshareDialogActive: true,
        });
        this.refresh();
    }

    setTitle(title) {
        this.update({
            mapshareTitle: title,
            mapshareCopied: false,
            mapsharePending: true,
        });
        this.refreshLater();
    }

    async refresh() {
        let params = {
            shape: this.shape,
            scale: this.scale,
            title: this.getValue('mapshareTitle') || '',
        };
        let n = ++this.requestCount;

        this.update({mapsharePending: true});

        let res = await this.app.server.call('mapshareCreateLink', params);

        if (n !== this.requestCount)
            return;

        this.update({
            mapshareUrl: res.error ? null : res.url,
            mapshareQrCode: res.error ? null : res.qrCode,
            mapsharePending: false,
        });
    }

    async copy() {
        try {
            await navigator.clipboard.writeText(this.getValue('mapshareUrl'));
            this.update({mapshareCopied: true});
        } catch (err) {
            console.log('MAPSHARE: copy failed', err);
        }
    }

    canShare() {
        return Boolean(navigator['share']);
    }

    async share() {
        let data = {
            title: (this.getValue('mapshareTitle') || '').trim() || document.title,
            url: this.getValue('mapshareUrl'),
        };
        try {
            await navigator['share'](data);
        } catch (err) {
            console.log('MAPSHARE: share failed', err);
        }
    }

    async showLink(link) {
        let res = await this.app.server.call('mapshareDecodeLink', {link});
        if (res.error) {
            console.log('MAPSHARE: invalid link', link);
            return;
        }

        let geometry = this.map.shape2geom(res.shape),
            feature = this.app.modelRegistry.defaultModel().featureFromGeometry(geometry),
            mode = 'draw zoom';

        if (res.scale) {
            this.map.setScale(res.scale);
            mode = 'draw pan';
        }

        this.update({
            marker: {
                features: [feature],
                mode,
            },
            infoboxContent: res.title ? <components.Infobox controller={this}>
                <p>{res.title}</p>
            </components.Infobox> : null,
        });
    }
}

class ToolbarButton extends toolbar.Button {
    iconClass = 'mapshareToolbarButton';
    tool = 'Tool.MapShare';

    get tooltip() {
        return this.__('mapshareToolbarButton');
    }
}

gc.registerTags({
    [MASTER]: Controller,
    'Toolbar.MapShare': ToolbarButton,
    'Tool.MapShare': MapShareTool,
});
