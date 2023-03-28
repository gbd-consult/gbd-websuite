import * as React from 'react';


import * as gws from 'gws';
import * as components from 'gws/components';
import * as toolbar from 'gws/elements/toolbar';
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
    printJob?: gws.api.base.printer.StatusResponse;
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
                // makeWidget={cc.makeWidget.bind(cc)}
                widget={cc.makeWidget(f, null, this.props.printFormData)}
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

        let w = gws.lib.mm2px(tpl.mapSize[0]);
        let h = gws.lib.mm2px(tpl.mapSize[1]);

        return null;

        // return <overview.SmallMap controller={this.props.controller} boxSize={[w, h]}/>;
    }

    optionsDialog() {
        let form = this.props.printSnapshotMode
            ? this.snapshotOptionsForm()
            : this.printOptionsForm();

        let ok = this.props.printSnapshotMode
            ? <gws.ui.Button
                {...gws.lib.cls('modPrintPreviewSnapshotButton')}
                whenTouched={() => this.props.controller.startSnapshot()}
                tooltip={this.__('modPrintPreviewSnapshotButton')}
            />
            : <gws.ui.Button
                {...gws.lib.cls('modPrintPreviewPrintButton')}
                whenTouched={() => this.props.controller.startPrinting()}
                tooltip={this.__('modPrintPreviewPrintButton')}
            />;

        return <div className="modPrintPreviewDialog">
            <Form>
                {form}
                <Row>
                    <Cell flex>
                        {this.overviewMap()}
                    </Cell>
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
                label = gws.lib.shorten(job.stepname, 40);


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
                {...gws.lib.cls('modPrintResultDialog', this.props.printDialogZoomed && 'isZoomed')}
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
        return null;
        // let cfg = field.widget;
        //
        // if (!cfg)
        //     return null;
        //
        // let type = cfg['type'];
        // let options = cfg['options'] || {};
        // let cls = gws.components.widget.WIDGETS[type];
        //
        // if (!cls)
        //     return null;
        //
        // let props = {
        //     controller: this,
        //     feature,
        //     field,
        //     values,
        //     options,
        //     readOnly: cfg.readOnly,
        //     whenChanged: this.whenFormChanged.bind(this),
        // }
        //
        // return React.createElement(cls, props);
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
            qualityLevel = Number(fd['_quality']) || 0,
            template = this.selectedTemplate,
            level = template.qualityLevels[qualityLevel],
            dpi = level ? level.dpi : 0;

        let basicParams = await this.map.basicPrintParams(
            this.previewBox.getBoundingClientRect(),
            dpi
        );

        let vs = this.map.viewState;

        let context = {...fd};
        delete context['_quality'];
        delete context['_templateIndex'];

        let params: gws.api.base.printer.ParamsWithTemplate = {
            type: 'template',
            templateUid: template.uid,
            qualityLevel,
            ...basicParams,
            // sections: [
            //     {
            //         center: [vs.centerX, vs.centerY] as gws.api.Point,
            //         context,
            //     }
            // ]
        };

        await this.startJob(this.app.server.printerStart(params, {binary: true}));
    }

    async startSnapshot() {
        let dpi = Number(this.getValue('printSnapshotDpi')) || MIN_SNAPSHOT_DPI;

        let basicParams = await this.map.basicPrintParams(
            this.previewBox.getBoundingClientRect(),
            dpi
        );

        let vs = this.map.viewState;

        let params: gws.api.base.printer.ParamsWithMap = {
            type: 'map',
            ...basicParams,
            // format: 'png',
            dpi: dpi,
            outputSize: [
                Number(this.getValue('printSnapshotWidth')) || DEFAULT_SNAPSHOT_SIZE,
                Number(this.getValue('printSnapshotHeight')) || DEFAULT_SNAPSHOT_SIZE,
            ],
            // sections: [
            //     {
            //         center: [vs.centerX, vs.centerY] as gws.api.Point,
            //     }
            // ]
        };

        await this.startJob(this.app.server.printerStart(params, {binary: true}));

    }

    async startJob(res) {
        this.update({
            printJob: {state: gws.api.lib.job.State.init}
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

            case gws.api.lib.job.State.init:
                this.update({printState: 'printing'});
                break;

            case gws.api.lib.job.State.open:
            case gws.api.lib.job.State.running:
                this.update({printState: 'printing'});
                this.jobTimer = setTimeout(() => this.poll(), JOB_POLL_INTERVAL);
                break;

            case gws.api.lib.job.State.cancel:
                this.stop();
                break;

            case gws.api.lib.job.State.complete:
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
            case gws.api.lib.job.State.error:
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
