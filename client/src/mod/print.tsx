import * as React from 'react';

import * as gws from 'gws';
import * as toolbar from './toolbar';
import * as overview from './overview';
import {FormField} from "gws/components/form";

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Printer';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as PrintController;

const JOB_POLL_INTERVAL = 2000;
const DEFAULT_SNAPSHOT_SIZE = 300;
const MIN_SNAPSHOT_DPI = 70;
const MAX_SNAPSHOT_DPI = 600;

interface PrintViewProps extends gws.types.ViewProps {
    controller: PrintController;
    printJob?: gws.api.JobStatusResponse;
    printState?: 'preview' | 'printing' | 'error' | 'complete';
    printFormData: object;
    printSnapshotMode: boolean;
    printSnapshotDpi: number;
    printSnapshotWidth: number;
    printSnapshotHeight: number;
    printDialogZoomed: boolean;
}

const PrintStoreKeys = [
    'printJob',
    'printState',
    'printFormData',
    'printSnapshotMode',
    'printSnapshotDpi',
    'printSnapshotWidth',
    'printSnapshotHeight',
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

    printOptionsForm() {
        let cc = this.props.controller;
        let fields = [];

        let templateItems = this.templateItems();
        if (templateItems.length > 1) {
            fields.push({
                type: "integer",
                name: "_templateIndex",
                title: this.__('modPrintTemplate'),
                widget: {
                    type: "select",
                    options: {
                        items: templateItems,
                    },
                    readOnly: false,
                }
            })
        }

        let qualityItems = this.qualityItems();
        if (qualityItems.length > 1) {
            fields.push({
                type: "integer",
                name: "_quality",
                title: this.__('modPrintQuality'),
                widget: {
                    type: "select",
                    options: {
                        items: qualityItems,
                    },
                    readOnly: false,
                }
            })
        }

        let tpl = cc.selectedTemplate;
        if (tpl && tpl.model)
            fields = fields.concat(tpl.model.fields)

        return <table className="cmpForm">
            <tbody>
            {fields.map((f, i) => <FormField
                key={i}
                field={f}
                controller={this.props.controller}
                feature={null}
                values={this.props.printFormData}
                makeWidget={cc.makeWidget.bind(cc)}
            />)}
            </tbody>
        </table>
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
                        label={this.__('modPrintSnapshotResolution')}
                        {...cc.bind('printSnapshotDpi')}
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    {this.props.printSnapshotDpi} dpi
                </Cell>

            </Row>
        </React.Fragment>
    }

    overviewMap() {
        let tpl = this.props.controller.selectedTemplate;

        if (!tpl)
            return null;

        let w = gws.tools.mm2px(tpl.mapWidth);
        let h = gws.tools.mm2px(tpl.mapHeight);

        return <overview.SmallMap controller={this.props.controller} boxSize={[w, h]}/>;
    }

    optionsDialog() {
        let form = this.props.printSnapshotMode
            ? this.snapshotOptionsForm()
            : this.printOptionsForm();

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
                {form}
                <Row>
                    <Cell flex>
                        {this.overviewMap()}
                    </Cell>
                </Row>
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

            let label = '';

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
                <gws.ui.TextBlock content={label}/>
            </gws.ui.Dialog>;
        }

        if (ps === 'complete') {
            return <gws.ui.Dialog
                {...gws.tools.cls('modPrintResultDialog', this.props.printDialogZoomed && 'isZoomed')}
                whenClosed={stop}
                whenZoomed={() => this.props.controller.update({printDialogZoomed: !this.props.printDialogZoomed})}
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
        let fd = this.getValue('printFormData') || {},
            idx = Number(fd['_templateIndex']) || 0;
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
        this.update({
            printSnapshotDpi: MIN_SNAPSHOT_DPI,
            printSnapshotWidth: DEFAULT_SNAPSHOT_SIZE,
            printSnapshotHeight: DEFAULT_SNAPSHOT_SIZE,
            printFormData: {
                _templateIndex: 0,
                _quality: 0,
            }
        })
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

    makeWidget(field: gws.types.IModelField, feature: gws.types.IFeature, values: gws.types.Dict): React.ReactElement | null {
        let cfg = field.widget;

        if (!cfg)
            return null;

        let type = cfg['type'];
        let options = cfg['options'] || {};
        let cls = gws.components.widget.WIDGETS[type];

        if (!cls)
            return null;

        let props = {
            controller: this,
            feature,
            field,
            values,
            options,
            readOnly: cfg.readOnly,
            whenChanged: this.whenFormChanged.bind(this),
        }

        return React.createElement(cls, props);
    }


    async whenFormChanged(field: gws.types.IModelField, value) {
        let fd = this.getValue('printFormData') || {};
        this.update({
            printFormData: {
                ...fd,
                [field.name]: value,
            }
        })
    }

    async startPrinting() {
        let fd = this.getValue('printFormData') || {},
            quality = Number(fd['_quality']) || 0,
            template = this.selectedTemplate,
            level = template.qualityLevels[quality],
            dpi = level ? level.dpi : 0;

        let basicParams = await this.map.basicPrintParams(
            this.previewBox.getBoundingClientRect(),
            dpi
        );

        let vs = this.map.viewState;

        let context = {...fd};
        delete context['_quality'];
        delete context['_templateIndex'];

        let params: gws.api.PrintParamsWithTemplate = {
            type: 'template',
            templateUid: template.uid,
            quality,
            ...basicParams,
            sections: [
                {
                    center: [vs.centerX, vs.centerY] as gws.api.Point,
                    context,
                }
            ]
        };

        await this.startJob(this.app.server.printerPrint(params, {binary: false}));
    }

    async startSnapshot() {
        let dpi = Number(this.getValue('printSnapshotDpi')) || MIN_SNAPSHOT_DPI;

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

        await this.startJob(this.app.server.printerSnapshot(params, {binary: false}));

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
                printJob: await this.app.server.printerStatus({jobUid: job.jobUid}),
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
