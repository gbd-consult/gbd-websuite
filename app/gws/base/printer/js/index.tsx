import * as React from 'react';

import * as gws from 'gws';
import * as components from 'gws/components';
import * as toolbar from 'gws/elements/toolbar';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Printer';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as Controller;

const JOB_POLL_INTERVAL = 2000;
const DEFAULT_SNAPSHOT_SIZE = 300;
const MIN_SNAPSHOT_DPI = 70;
const MAX_SNAPSHOT_DPI = 600;

interface ViewProps extends gws.types.ViewProps {
    controller: Controller;
    printerJob?: gws.api.base.printer.StatusResponse;
    printerQuality: string;
    printerState?: 'preview' | 'running' | 'error' | 'complete';
    printerTemplateIndex: string;
    printerData: object;
    printerSnapshotMode: boolean;
    printerSnapshotDpi: number;
    printerSnapshotWidth: number;
    printerSnapshotHeight: number;
    printDialogZoomed: boolean;
}

const StoreKeys = [
    'printerJob',
    'printerQuality',
    'printerState',
    'printerTemplateIndex',
    'printerData',
    'printerSnapshotMode',
    'printerSnapshotDpi',
    'printerSnapshotWidth',
    'printerSnapshotHeight',
    'printDialogZoomed',
];

class PrintTool extends gws.Tool {
    start() {
        _master(this).startPrintPreview()
    }

    stop() {
        _master(this).reset();
    }
}

class SnapshotTool extends gws.Tool {
    start() {
        _master(this).startSnapshotPreview()
    }

    stop() {
        _master(this).reset();
    }
}

class PrintToolbarButton extends toolbar.Button {
    iconClass = 'printerPrintToolbarButton';
    tool = 'Tool.Printer.Print';

    get tooltip() {
        return this.__('printerPrintToolbarButton');
    }
}

class SnapshotToolbarButton extends toolbar.Button {
    iconClass = 'printerSnapshotToolbarButton';
    tool = 'Tool.Printer.Snapshot';

    get tooltip() {
        return this.__('printerSnapshotToolbarButton');
    }
}

class PreviewBox extends gws.View<ViewProps> {

    boxRef: React.RefObject<HTMLDivElement>;

    constructor(props) {
        super(props);
        this.boxRef = React.createRef();
    }

    componentDidUpdate() {
        this.props.controller.previewBox = this.boxRef.current;
    }

    templateItems() {
        return this.props.controller.templates.map((t, n) => ({
            value: String(n),
            text: t.title,
        }));
    }

    qualityItems() {
        let tpl = this.props.controller.selectedTemplate;
        if (!tpl)
            return [];

        return (tpl.qualityLevels || []).map((level, n) => ({
            value: String(n),
            text: level.name || (level.dpi + ' dpi')
        }));
    }

    printOptionsForm() {
        let data = [], items;

        items = this.templateItems();
        if (items.length > 1) {
            data.push({
                name: 'printerTemplateIndex',
                title: this.__('printerTemplate'),
                value: String(this.props.printerTemplateIndex || 0),
                editable: true,
                type: 'select',
                items
            })
        }

        items = this.qualityItems();
        if (items.length > 1) {
            data.push({
                name: 'printerQuality',
                title: this.__('printerQuality'),
                value: this.props.printerQuality || '0',
                editable: true,
                type: 'select',
                items
            })
        }

        let pd = this.props.printerData || {};
        let tpl = this.props.controller.selectedTemplate;

        // if (tpl && tpl.dataModel) {
        //     tpl.dataModel.rules.forEach(r => data.push({
        //         name: r.name,
        //         title: r.title || r.name,
        //         value: pd[r.name] || '',
        //         editable: true,
        //         type: r.type,
        //     }));
        // }

        let changed = (k, v) => {
            let up;
            if (k == 'printerTemplateIndex' || k == 'printerQuality')
                up = {[k]: v}
            else
                up = {
                    printerData: {...pd, [k]: v}
                };

            this.props.controller.update(up)
        };

        return <Row><Cell>
            <components.sheet.Editor
                data={data}
                whenChanged={changed}
            />
        </Cell></Row>;
    }

    snapshotOptionsForm() {
        let cc = this.props.controller;

        return <React.Fragment>

            <Row>
                <Cell flex>
                    <gws.ui.Slider
                        minValue={MIN_SNAPSHOT_DPI}
                        maxValue={MAX_SNAPSHOT_DPI}
                        step={10}
                        label={this.__('printerSnapshotResolution')}
                        {...cc.bind('printerSnapshotDpi')}
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    {this.props.printerSnapshotDpi} dpi
                </Cell>

            </Row>
        </React.Fragment>
    }

    optionsDialog() {
        let form = this.props.printerSnapshotMode
            ? this.snapshotOptionsForm()
            : this.printOptionsForm();

        let ok = this.props.printerSnapshotMode
            ? <gws.ui.Button
                {...gws.lib.cls('printerPreviewSnapshotButton')}
                whenTouched={() => this.props.controller.startSnapshot()}
                tooltip={this.__('printerPreviewSnapshotButton')}
            />
            : <gws.ui.Button
                {...gws.lib.cls('printerPreviewPrintButton')}
                whenTouched={() => this.props.controller.startPrinting()}
                tooltip={this.__('printerPreviewPrintButton')}
            />;

        return <div className="printerPreviewDialog">
            <Form>
                {form}
                <Row>
                    <Cell flex/>
                    <Cell>
                        {ok}
                    </Cell>
                    <Cell>
                        <gws.ui.Button
                            className="cmpButtonFormCancel"
                            whenTouched={() => this.props.controller.app.startTool('Tool.Default')}
                            tooltip={this.__('printerCancel')}
                        />
                    </Cell>
                </Row>
            </Form>
        </div>
    }

    handleTouched() {

        gws.lib.trackDrag({
            map: this.props.controller.map,
            whenMoved: px => {
                let sz = this.props.controller.map.oMap.getSize();
                let w = Math.max(50, 2 * Math.abs(px[0] - (sz[0] >> 1)));
                let h = Math.max(50, 2 * Math.abs(px[1] - (sz[1] >> 1)))

                this.props.controller.update({
                    printerSnapshotWidth: w,
                    printerSnapshotHeight: h,
                });

            },
        })
    }

    render() {
        let ps = this.props.printerState;

        if (ps !== 'preview')
            return null;

        let w, h;

        if (this.props.printerSnapshotMode) {

            w = this.props.printerSnapshotWidth || DEFAULT_SNAPSHOT_SIZE;
            h = this.props.printerSnapshotHeight || DEFAULT_SNAPSHOT_SIZE;

        } else {

            let tpl = this.props.controller.selectedTemplate;

            if (!tpl)
                return null;

            w = gws.lib.mm2px(tpl.mapSize[0]);
            h = gws.lib.mm2px(tpl.mapSize[1]);
        }

        let style = {
            width: w,
            height: h,
            marginLeft: -(w >> 1),
            marginTop: -(h >> 1)
        };

        const handleSize = 40;

        return <div>
            <div className="printerPreviewBox" ref={this.boxRef} style={style}/>
            {this.props.printerSnapshotMode && <div
                className="printerPreviewBoxHandle"
                style={{
                    marginLeft: -(w >> 1) - (handleSize >> 1),
                    marginTop: -(h >> 1) - (handleSize >> 1),

                }}
                onMouseDown={() => this.handleTouched()}
            />
            }

            {this.optionsDialog()}
        </div>;

    }

}

class ProgressDialog extends gws.View<ViewProps> {

    render() {
        let ps = this.props.printerState;
        let job = this.props.printerJob;

        let cancel = () => this.props.controller.cancelPrinting();
        let stop = () => this.props.controller.stop();

        if (ps === 'running') {

            let label = '';

            if (job.steptype === 'begin_plane' && job.stepname)
                label = gws.lib.shorten(job.stepname, 40);

            if (job.steptype === 'begin_page' && job.stepname)
                label = gws.lib.shorten(job.stepname, 40);


            return <gws.ui.Dialog
                className="printerProgressDialog"
                title={this.__('printerPrinting')}
                whenClosed={cancel}
                buttons={[
                    <gws.ui.Button label={this.__('printerCancel')} whenTouched={cancel}/>
                ]}
            >
                <gws.ui.Progress value={job.progress}/>
                <gws.ui.TextBlock content={label}/>
            </gws.ui.Dialog>;
        }

        if (ps === 'complete') {
            return <gws.ui.Dialog
                {...gws.lib.cls('printerResultDialog', this.props.printDialogZoomed && 'isZoomed')}
                whenClosed={stop}
                whenZoomed={() => this.props.controller.update({printDialogZoomed: !this.props.printDialogZoomed})}
                frame={job.url}
            />;
        }

        if (ps === 'error') {
            return <gws.ui.Alert
                whenClosed={stop}
                title={this.__('appError')}
                error={this.__('printerError')}
            />;
        }

        return null;
    }
}

class Controller extends gws.Controller {
    uid = MASTER;
    jobTimer: any = null;
    previewBox: HTMLDivElement;

    get templates() {
        if (!this.app.project.printer)
            return [];
        // return this.app.project.printer.templates.items || [];
        return [];
    }

    get selectedTemplate() {
        let idx = Number(this.getValue('printerTemplateIndex') || 0);
        return this.templates[idx] || this.templates[0];
    }

    get mapOverlayView() {
        return this.createElement(
            this.connect(PreviewBox, StoreKeys));
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(ProgressDialog, StoreKeys));
    }

    get activeJob() {
        return this.getValue('printerJob');
    }

    async init() {
        await super.init();
        this.app.whenChanged('printerJob', job => this.jobUpdated(job));
        this.update({
            printerSnapshotDpi: MIN_SNAPSHOT_DPI,
            printerSnapshotWidth: DEFAULT_SNAPSHOT_SIZE,
            printerSnapshotHeight: DEFAULT_SNAPSHOT_SIZE
        })
    }

    reset() {
        this.update({
            printerJob: null,
            printerState: null,
            toolbarItem: null,
        });
        clearTimeout(this.jobTimer);
        this.jobTimer = 0;
    }

    stop() {
        this.app.stopTool('Tool.Print.*');
        this.reset();
    }

    startPrintPreview() {
        this.map.resetInteractions();
        return this.update({
            printerState: 'preview',
            printerSnapshotMode: false,
        });
    }

    startSnapshotPreview() {
        this.map.resetInteractions();
        return this.update({
            printerState: 'preview',
            printerSnapshotMode: true,
        });
    }

    async startPrinting() {
        let qualityLevel = Number(this.getValue('printerQuality')) || 0,
            level = this.selectedTemplate.qualityLevels[qualityLevel],
            dpi = level ? level.dpi : 0;

        let mapParams = await this.map.printParams(
            this.previewBox.getBoundingClientRect(),
            dpi
        );

        let vs = this.map.viewState;
        mapParams.center = [vs.centerX, vs.centerY] as gws.api.core.Point;

        let params: gws.api.base.printer.ParamsWithTemplate = {
            context: this.getValue('printerData'),
            maps: [mapParams],
            qualityLevel,
            templateUid: this.selectedTemplate.uid,
            type: 'template',
        };

        await this.startJob(this.app.server.printerStart(params, {binary: true}));
    }

    async startSnapshot() {
        let dpi = Number(this.getValue('printerSnapshotDpi')) || MIN_SNAPSHOT_DPI;

        let mapParams = await this.map.printParams(
            this.previewBox.getBoundingClientRect(),
            dpi
        );

        let w = Number(this.getValue('printerSnapshotWidth')) || DEFAULT_SNAPSHOT_SIZE;
        let h = Number(this.getValue('printerSnapshotHeight')) || DEFAULT_SNAPSHOT_SIZE;

        let vs = this.map.viewState;
        let res = vs.resolution;

        mapParams.bbox = [
            vs.centerX - (w / 2) * res,
            vs.centerY - (h / 2) * res,
            vs.centerX + (w / 2) * res,
            vs.centerY + (h / 2) * res,
        ];

        let params: gws.api.base.printer.ParamsWithMap = {
            context: this.getValue('printerData'),
            dpi,
            maps: [mapParams],
            outputFormat: 'png',
            outputSize: [w, h],
            type: 'map',
        };

        await this.startJob(this.app.server.printerStart(params, {binary: true}));

    }

    async startJob(res) {
        this.update({
            printerJob: {state: gws.api.lib.job.State}
        });

        let job = await res;

        // the job can be canceled in the meantime

        if (!this.activeJob) {
            console.log('JOB ALREADY CANCELED')
            this.sendCancel(job.jobUid);
            return;
        }

        this.update({
            printerJob: job
        });

    }

    async cancelPrinting() {
        let job = this.activeJob,
            jobUid = job ? job.jobUid : null;

        this.stop();
        await this.sendCancel(jobUid);
    }

    protected jobUpdated(job) {
        if (!job) {
            return this.update({printerState: null});
        }

        if (job.error) {
            return this.update({printerState: 'error'});
        }

        console.log('JOB_UPDATED', job.state);

        switch (job.state) {

            case gws.api.lib.job.State.init:
                this.update({printerState: 'running'});
                break;

            case gws.api.lib.job.State.open:
            case gws.api.lib.job.State.running:
                this.update({printerState: 'running'});
                this.jobTimer = setTimeout(() => this.poll(), JOB_POLL_INTERVAL);
                break;

            case gws.api.lib.job.State.cancel:
                this.stop();
                break;

            case gws.api.lib.job.State.complete:
                if (this.getValue('printerSnapshotMode')) {
                    let a = document.createElement('a');
                    a.href = job.url;
                    a.download = 'image.png';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    this.stop()
                } else {
                    this.update({printerState: job.state});
                }
                break;
            case gws.api.lib.job.State.error:
                this.update({printerState: job.state});
        }
    }

    protected async poll() {
        let job = this.getValue('printerJob');

        if (job) {
            this.update({
                printerJob: await this.app.server.printerStatus({jobUid: job.jobUid}),
            });
        }
    }

    protected async sendCancel(jobUid) {
        if (jobUid) {
            console.log('SEND CANCEL');
            await this.app.server.printerCancel({jobUid});
        }
    }

}

gws.registerTags({
    [MASTER]: Controller,
    'Toolbar.Print': PrintToolbarButton,
    'Toolbar.Snapshot': SnapshotToolbarButton,
    'Tool.Printer.Print': PrintTool,
    'Tool.Printer.Snapshot': SnapshotTool,
});
