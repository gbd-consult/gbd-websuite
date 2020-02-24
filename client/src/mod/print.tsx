import * as React from 'react';

import * as gws from 'gws';
import * as toolbar from './common/toolbar';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Printer';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as PrintController;

const JOB_POLL_INTERVAL = 2000;
const DEFAULT_SNAPSHOT_SIZE = 300;

interface PrintViewProps extends gws.types.ViewProps {
    controller: PrintController;
    printJob?: gws.api.PrinterResponse;
    printQuality: string;
    printState?: 'preview' | 'printing' | 'error' | 'complete';
    printTemplateIndex: string;
    printData: object;
    printSnapshotMode: boolean;
    printSnapshotDpi: number;
    printSnapshotWidth: number;
    printSnapshotHeight: number;
}

const PrintStoreKeys = [
    'printJob',
    'printQuality',
    'printState',
    'printTemplateIndex',
    'printData',
    'printSnapshotMode',
    'printSnapshotDpi',
    'printSnapshotWidth',
    'printSnapshotHeight',
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

class PrintPrintToolbarButton extends toolbar.Button {
    iconClass = 'modPrintPrintToolbarButton';
    tool = 'Tool.Print.Print';

    get tooltip() {
        return this.__('modPrintPrintToolbarButton');
    }
}

class PrintSnapshotToolbarButton extends toolbar.Button {
    iconClass = 'modPrintSnapshotToolbarButton';
    tool = 'Tool.Print.Snapshot';

    get tooltip() {
        return this.__('modPrintSnapshotToolbarButton');
    }
}

class PrintPreviewBox extends gws.View<PrintViewProps> {

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
            text: t.title
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

    printDataSheet() {
        let data = [], items;

        items = this.templateItems();
        if (items.length > 1) {
            data.push({
                name: 'printTemplateIndex',
                title: this.__('modPrintTemplate'),
                value: String(this.props.printTemplateIndex || 0),
                editable: true,
                type: 'select',
                items
            })
        }

        items = this.qualityItems();
        if (items.length > 1) {
            data.push({
                name: 'printQuality',
                title: this.__('modPrintQuality'),
                value: this.props.printQuality || '0',
                editable: true,
                type: 'select',
                items
            })
        }

        let pd = this.props.printData || {};
        let tpl = this.props.controller.selectedTemplate;

        if (tpl && tpl.dataModel) {
            tpl.dataModel.rules.forEach(r => data.push({
                name: r.name,
                title: r.title || r.name,
                value: pd[r.name] || '',
                editable: true,
                type: r.type,
            }));
        }

        let changed = (k, v) => {
            let up;
            if (k == 'printTemplateIndex' || k == 'printQuality')
                up = {[k]: v}
            else
                up = {
                    printData: {...pd, [k]: v}
                };

            this.props.controller.update(up)
        };

        return <gws.components.sheet.Editor
            data={data}
            whenChanged={changed}
        />;
    }

    snapshotDataSheet() {
        let data = [];

        let changed = (k, v) => {
            this.props.controller.update({[k]: v})
        };

        data.push({
            name: 'printSnapshotDpi',
            title: this.__('modPrintDpi'),
            value: this.props.printSnapshotDpi || '',
            editable: true,
        });

        return <gws.components.sheet.Editor
            data={data}
            whenChanged={changed}
        />;

    }

    optionsDialog() {
        let vs = this.props.controller.map.viewState;

        let ds = this.props.printSnapshotMode
            ? this.snapshotDataSheet()
            : this.printDataSheet();

        let ok = this.props.printSnapshotMode
            ? <gws.ui.Button
                {...gws.tools.cls('modPrintPreviewSnapshotButton')}
                whenTouched={() => this.props.controller.startSnapshot()}
                tooltip={this.__('modPrintPreviewSnapshotButton')}
            />
            : <gws.ui.Button
                {...gws.tools.cls('modPrintPreviewPrintButton')}
                whenTouched={() => this.props.controller.startPrinting()}
                tooltip={this.__('modPrintPreviewPrintButton')}
            />;

        return <div className="modPrintPreviewDialog">
            <Form>
                <Row>
                    <Cell flex>{ds}</Cell>
                </Row>
                <Row>
                    <Cell flex/>
                    <Cell>
                        {ok}
                    </Cell>
                    <Cell>
                        <gws.ui.Button
                            className="cmpButtonFormCancel"
                            whenTouched={() => this.props.controller.app.startTool('Tool.Default')}
                            tooltip={this.__('modPrintCancel')}
                        />
                    </Cell>
                </Row>
                {vs.rotation !== 0 && <gws.ui.Error text={this.__('modPrintRotationWarning')}/>}
            </Form>
        </div>
    }

    handleTouched() {

        gws.tools.trackDrag({
            map: this.props.controller.map,
            whenMoved: px => {
                let sz = this.props.controller.map.oMap.getSize();
                let w = Math.max(50, 2 * Math.abs(px[0] - (sz[0] >> 1)));
                let h = Math.max(50, 2 * Math.abs(px[1] - (sz[1] >> 1)))

                this.props.controller.update({
                    printSnapshotWidth: w,
                    printSnapshotHeight: h,
                });

            },
        })
    }

    render() {
        let ps = this.props.printState;

        if (ps !== 'preview')
            return null;

        let w, h;

        if (this.props.printSnapshotMode) {

            w = this.props.printSnapshotWidth || DEFAULT_SNAPSHOT_SIZE;
            h = this.props.printSnapshotHeight || DEFAULT_SNAPSHOT_SIZE;

        } else {

            let tpl = this.props.controller.selectedTemplate;

            if (!tpl)
                return null;

            w = gws.tools.mm2px(tpl.mapWidth);
            h = gws.tools.mm2px(tpl.mapHeight);
        }

        let style = {
            width: w,
            height: h,
            marginLeft: -(w >> 1),
            marginTop: -(h >> 1)
        };

        const handleSize = 40;

        return <div>
            <div className="modPrintPreviewBox" ref={this.boxRef} style={style}/>
            {this.props.printSnapshotMode && <div
                className="modPrintPreviewBoxHandle"
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

class PrintDialog extends gws.View<PrintViewProps> {

    render() {
        let ps = this.props.printState;
        let job = this.props.printJob;

        let cancel = () => this.props.controller.cancelPrinting();
        let stop = () => this.props.controller.stop();

        if (ps === 'printing') {

            let label =  '';

            if (job.steptype === 'layer' && job.stepname)
                label = gws.tools.shorten(job.stepname, 40);


            return <gws.ui.Dialog
                className="modPrintProgressDialog"
                title={this.__('modPrintPrinting')}
                whenClosed={cancel}
                buttons={[
                    <gws.ui.Button label={this.__('modPrintCancel')} whenTouched={cancel}/>
                ]}
            >
                <gws.ui.Progress value={job.progress}/>
                <gws.ui.TextBlock content={label} />
            </gws.ui.Dialog>;
        }

        if (ps === 'complete') {
            return <gws.ui.Dialog
                className="modPrintResultDialog"
                whenClosed={stop}
                frame={job.url}
            />;
        }

        if (ps === 'error') {
            return <gws.ui.Alert
                whenClosed={stop}
                title={this.__('appError')}
                error={this.__('modPrintError')}
            />;
        }

        return null;
    }
}

class PrintController extends gws.Controller {
    uid = MASTER;
    jobTimer: any = null;
    previewBox: HTMLDivElement;

    get templates() {
        if (!this.app.project.printer)
            return [];
        return this.app.project.printer.templates || [];
    }

    get selectedTemplate() {
        let idx = Number(this.getValue('printTemplateIndex') || 0);
        return this.templates[idx] || this.templates[0];
    }

    get mapOverlayView() {
        return this.createElement(
            this.connect(PrintPreviewBox, PrintStoreKeys));
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(PrintDialog, PrintStoreKeys));
    }

    get activeJob() {
        return this.getValue('printJob');
    }

    async init() {
        await super.init();
        this.app.whenChanged('printJob', job => this.jobUpdated(job));
    }

    reset() {
        this.update({
            printJob: null,
            printState: null,
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
            printState: 'preview',
            printSnapshotMode: false,
        });
    }

    startSnapshotPreview() {
        this.map.resetInteractions();
        return this.update({
            printState: 'preview',
            printSnapshotMode: true,
        });
    }

    async startPrinting() {
        let quality = Number(this.getValue('printQuality')) || 0,
            level = this.selectedTemplate.qualityLevels[quality],
            dpi = level ? level.dpi : 0;

        let basicParams = await this.map.basicPrintParams(
            this.previewBox.getBoundingClientRect(),
            dpi
        );

        let vs = this.map.viewState;

        let params: gws.api.PrintParamsWithTemplate = {
            type: 'template',
            templateUid: this.selectedTemplate.uid,
            quality,
            ...basicParams,
            sections: [
                {
                    center: [vs.centerX, vs.centerY] as gws.api.Point,
                    context: this.getValue('printData')
                }
            ]
        };

        await this.startJob(this.app.server.printerPrint(params, {binary: true}));
    }

    async startSnapshot() {
        let dpi = Number(this.getValue('printSnapshotDpi')) || 0;

        let basicParams = await this.map.basicPrintParams(
            this.previewBox.getBoundingClientRect(),
            dpi
        );

        let vs = this.map.viewState;

        let params: gws.api.PrintParamsWithMap = {
            type: 'map',
            ...basicParams,
            format: 'png',
            dpi: dpi,
            mapWidth: Number(this.getValue('printSnapshotWidth')) || DEFAULT_SNAPSHOT_SIZE,
            mapHeight: Number(this.getValue('printSnapshotHeight')) || DEFAULT_SNAPSHOT_SIZE,
            sections: [
                {
                    center: [vs.centerX, vs.centerY] as gws.api.Point,
                }
            ]
        };

        await this.startJob(this.app.server.printerSnapshot(params, {binary: true}));

    }

    async startJob(res) {
        this.update({
            printJob: {state: gws.api.JobState.init}
        });

        let job = await res;

        // the job can be canceled in the meantime

        if (!this.activeJob) {
            console.log('JOB ALREADY CANCELED')
            this.sendCancel(job.jobUid);
            return;
        }

        this.update({
            printJob: job
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
            return this.update({printState: null});
        }

        if (job.error) {
            return this.update({printState: 'error'});
        }

        console.log('JOB_UPDATED', job.state);

        switch (job.state) {

            case gws.api.JobState.init:
                this.update({printState: 'printing'});
                break;

            case gws.api.JobState.open:
            case gws.api.JobState.running:
                this.update({printState: 'printing'});
                this.jobTimer = setTimeout(() => this.poll(), JOB_POLL_INTERVAL);
                break;

            case gws.api.JobState.cancel:
                this.stop();
                break;

            case gws.api.JobState.complete:
                if (this.getValue('printSnapshotMode')) {
                    let a = document.createElement('a');
                    a.href = job.url;
                    a.download = 'image.png';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    this.stop()
                } else {
                    this.update({printState: job.state});
                }
                break;
            case gws.api.JobState.error:
                this.update({printState: job.state});
        }
    }

    protected async poll() {
        let job = this.getValue('printJob');

        if (job) {
            this.update({
                printJob: await this.app.server.printerQuery({jobUid: job.jobUid}),
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

export const tags = {
    [MASTER]: PrintController,
    'Toolbar.Print': PrintPrintToolbarButton,
    'Toolbar.Snapshot': PrintSnapshotToolbarButton,
    'Tool.Print.Print': PrintTool,
    'Tool.Print.Snapshot': SnapshotTool,
};
