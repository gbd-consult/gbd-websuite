import * as React from 'react';


import * as gws from 'gws';
import * as overview from 'gws/elements/overview';
import * as toolbar from 'gws/elements/toolbar';
import {FormField} from "gws/components/form";

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Printer';

function _master(obj: any): Controller {
    if (obj.app)
        return obj.app.controller(MASTER) as Controller;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as Controller;
}

const JOB_POLL_INTERVAL = 2000;
const DEFAULT_SNAPSHOT_SIZE = 300;
const MIN_SNAPSHOT_DPI = 70;
const MAX_SNAPSHOT_DPI = 600;

interface ViewProps extends gws.types.ViewProps {
    controller: Controller;
    printerDialogZoomed: boolean;
    printerFormValues: object;
    printerJob?: gws.api.base.printer.StatusResponse;
    printerMode: 'screenshot' | 'print';
    printerScreenshotDpi: number;
    printerScreenshotHeight: number;
    printerScreenshotWidth: number;
    printerState?: 'preview' | 'printing' | 'error' | 'complete';
}

const StoreKeys = [
    'printerDialogZoomed',
    'printerFormValues',
    'printerJob',
    'printerMode',
    'printerScreenshotDpi',
    'printerScreenshotHeight',
    'printerScreenshotWidth',
    'printerState',
];

class PrintTool extends gws.Tool {
    start() {
        _master(this).startPrintPreview()
    }

    stop() {
        _master(this).reset();
    }
}

class ScreenshotTool extends gws.Tool {
    start() {
        _master(this).startScreenshotPreview()
    }

    stop() {
        _master(this).reset();
    }
}

class PrintToolbarButton extends toolbar.Button {
    iconClass = 'printerPrintToolbarButton';
    tool = 'Tool.Print.Print';

    get tooltip() {
        return this.__('printerPrintToolbarButton');
    }
}

class ScreenshotToolbarButton extends toolbar.Button {
    iconClass = 'printerScreenshotToolbarButton';
    tool = 'Tool.Print.Screenshot';

    get tooltip() {
        return this.__('printerScreenshotToolbarButton');
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

    printOptionsForm() {
        let cc = _master(this);
        let prt = cc.selectedPrinter;

        let fields = [];
        let values = this.props.printerFormValues || {};

        let printerItems = cc.printers.map((p, n) => ({
            value: n,
            text: p.title,
        }))

        if (printerItems.length > 1) {
            fields.push({
                type: "integer",
                name: "_printerIndex",
                title: this.__('printerTemplate'),
                widgetProps: {
                    type: "select",
                    items: printerItems,
                    readOnly: false,
                }
            })
        }

        let qualityItems = [];
        if (prt) {
            qualityItems = (prt.qualityLevels || []).map((level, n) => ({
                value: n,
                text: level.name || (level.dpi + ' dpi')
            }));
        }

        if (qualityItems.length > 1) {
            fields.push({
                type: "integer",
                name: "_qualityIndex",
                title: this.__('printerQuality'),
                widgetProps: {
                    type: "select",
                    items: qualityItems,
                    readOnly: false,
                }
            })
        }


        if (prt && prt.model) {
            fields = fields.concat(prt.model.fields)
        }

        if (fields.length === 0)
            return null;

        return <Row>
            <Cell flex>
                <table className="cmpForm">
                    <tbody>
                    {fields.map((f, i) => <FormField
                        key={i}
                        field={f}
                        controller={this.props.controller}
                        feature={null}
                        values={this.props.printerFormValues}
                        widget={cc.createWidget(f, values)}
                    />)}
                    </tbody>
                </table>
            </Cell>
        </Row>
    }

    screenshotOptionsForm() {
        let cc = _master(this);

        return <React.Fragment>
            <Row>
                <Cell flex>
                    <gws.ui.Slider
                        minValue={MIN_SNAPSHOT_DPI}
                        maxValue={MAX_SNAPSHOT_DPI}
                        step={10}
                        label={this.__('printerScreenshotResolution')}
                        {...cc.bind('printerScreenshotDpi')}
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    {this.props.printerScreenshotDpi} dpi
                </Cell>

            </Row>
        </React.Fragment>
    }

    overviewMap() {
        let cc = _master(this);
        let prt = cc.selectedPrinter;

        if (!cc.app.project.overviewMap)
            return null;

        if (!prt)
            return null;

        let w = gws.lib.mm2px(prt.template.mapSize[0]);
        let h = gws.lib.mm2px(prt.template.mapSize[1]);

        return <Row>
            <Cell flex>
                <overview.SmallMap controller={this.props.controller} boxSize={[w, h]}/>
            </Cell>
        </Row>;
    }

    optionsDialog() {
        let form = (this.props.printerMode === 'screenshot')
            ? this.screenshotOptionsForm()
            : this.printOptionsForm();

        let ok = (this.props.printerMode === 'screenshot')
            ? <gws.ui.Button
                {...gws.lib.cls('printerPreviewScreenshotButton')}
                whenTouched={() => this.props.controller.whenPreviewOkButtonTouched()}
                tooltip={this.__('printerPreviewScreenshotButton')}
            />
            : <gws.ui.Button
                {...gws.lib.cls('printerPreviewPrintButton')}
                whenTouched={() => this.props.controller.whenPreviewOkButtonTouched()}
                tooltip={this.__('printerPreviewPrintButton')}
            />;

        let map = this.overviewMap()

        return <div className="printerPreviewDialog">
            <Form>
                <Row>
                    <Cell flex/>
                    <Cell>
                        {ok}
                    </Cell>
                    <Cell>
                        <gws.ui.Button
                            className="cmpButtonFormCancel"
                            whenTouched={() => this.props.controller.whenPreviewCancelButtonTouched()}
                            tooltip={this.__('printerCancel')}
                        />
                    </Cell>
                </Row>
                {form}
                {map}
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
                    printerScreenshotWidth: w,
                    printerScreenshotHeight: h,
                });

            },
        })
    }

    render() {
        let cc = _master(this);
        let prt = cc.selectedPrinter;

        let ps = this.props.printerState;

        if (ps !== 'preview')
            return null;

        let w, h;

        if (this.props.printerMode === 'screenshot') {

            w = this.props.printerScreenshotWidth || DEFAULT_SNAPSHOT_SIZE;
            h = this.props.printerScreenshotHeight || DEFAULT_SNAPSHOT_SIZE;

        } else {
            if (!prt)
                return null;
            w = gws.lib.mm2px(prt.template.mapSize[0]);
            h = gws.lib.mm2px(prt.template.mapSize[1]);
        }

        let style = {
            width: w,
            height: h,
            marginLeft: -(w >> 1),
            marginTop: -(h >> 1)
        };

        const handleSize = 40;

        let handle = <div
            className="printerPreviewBoxHandle"
            style={{
                marginLeft: -(w >> 1) - (handleSize >> 1),
                marginTop: -(h >> 1) - (handleSize >> 1),

            }}
            onMouseDown={() => this.handleTouched()}
        />

        return <div>
            <div className="printerPreviewBox" ref={this.boxRef} style={style}/>
            {(this.props.printerMode === 'screenshot') && handle}
            {this.optionsDialog()}
        </div>;

    }

}

class Dialog extends gws.View<ViewProps> {

    render() {
        let cc = _master(this);
        let ps = this.props.printerState;
        let job = this.props.printerJob;

        let cancel = () => cc.cancelPrinting();
        let stop = () => cc.stop();

        if (ps === 'printing') {

            let label = '';

            if (job.stepName)
                label = gws.lib.shorten(job.stepName, 40);

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
                {...gws.lib.cls('printerResultDialog')}
                whenClosed={stop}
                frame={job.url}
            />;
            // return <gws.ui.Dialog
            //     {...gws.lib.cls('printerResultDialog', this.props.printerDialogZoomed && 'isZoomed')}
            //     whenClosed={stop}
            //     whenZoomed={() => cc.update({printerDialogZoomed: !this.props.printerDialogZoomed})}
            //     frame={job.url}
            // />;
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

class Printer {
    uid: string
    template: gws.api.base.template.Props
    qualityLevels: Array<gws.api.core.TemplateQualityLevel>
    title: string
    model?: gws.types.IModel
}

class Controller extends gws.Controller {
    uid = MASTER;
    jobTimer: any = null;
    previewBox: HTMLDivElement;
    printers: Array<Printer>;


    get selectedPrinter(): Printer {
        let fd = this.getValue('printerFormValues') || {},
            idx = Number(fd['_printerIndex']) || 0;
        return this.printers[idx] || this.printers[0];
    }

    get mapOverlayView() {
        return this.createElement(
            this.connect(PreviewBox, StoreKeys));
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(Dialog, StoreKeys));
    }

    get activeJob() {
        return this.getValue('printerJob');
    }

    async init() {
        await super.init();

        this.printers = [];

        if (this.app.project.printers) {
            for (let p of this.app.project.printers) {
                let prt = new Printer();
                prt.uid = p.uid;
                prt.template = p.template;
                prt.qualityLevels = p.qualityLevels;
                prt.title = p.title;
                if (p.model)
                    prt.model = this.app.modelRegistry.readModel(p.model)
                this.printers.push(prt);
            }
        }

        this.app.whenChanged('printerJob', job => this.jobUpdated(job));

        this.update({
            printerScreenshotDpi: MIN_SNAPSHOT_DPI,
            printerScreenshotWidth: DEFAULT_SNAPSHOT_SIZE,
            printerScreenshotHeight: DEFAULT_SNAPSHOT_SIZE,
            printerFormValues: {
                _printerIndex: 0,
                _qualityIndex: 0,
            }
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
            printerMode: 'print',
        });
    }

    startScreenshotPreview() {
        this.map.resetInteractions();
        return this.update({
            printerState: 'preview',
            printerMode: 'screenshot',
        });
    }

    createWidget(field: gws.types.IModelField, values: gws.types.Dict): React.ReactElement | null {
        let p = field.widgetProps;

        if (!p)
            return null;

        let tag = 'ModelWidget.' + p.type;
        let controller = (this.app.controllerByTag(tag) || this.app.createControllerFromConfig(this, {tag})) as gws.types.IModelWidget;

        let props: gws.types.Dict = {
            controller,
            field,
            widgetProps: field.widgetProps,
            values,
            whenChanged: val => this.whenWidgetChanged(field, val),
            whenEntered: val => this.whenWidgetEntered(field, val),
        }


        return controller.formView(props)
    }

    whenWidgetChanged(field: gws.types.IModelField, value) {
        let fd = this.getValue('printerFormValues') || {};
        this.update({
            printerFormValues: {
                ...fd,
                [field.name]: value,
            }
        })
    }

    whenWidgetEntered(field: gws.types.IModelField, value) {
        this.whenPreviewOkButtonTouched()
    }

    whenPreviewOkButtonTouched() {
        if (this.getValue('printerMode') === 'print')
            return this.startPrint()
        if (this.getValue('printerMode') === 'screenshot')
            return this.startScreenshot()
    }

    whenPreviewCancelButtonTouched() {
        this.stop()
    }

    //

    async startPrint() {
        let prt = this.selectedPrinter;
        let fd = this.getValue('printerFormValues') || {};

        let level = prt.qualityLevels[Number(fd['_qualityIndex']) || 0];
        let dpi = level ? level.dpi : 0;

        let mapParams = await this.map.printParams(
            this.previewBox.getBoundingClientRect(),
            dpi
        );

        // @TODO should create a new feature for the printer model

        let args = {...fd};
        delete args['_qualityIndex'];
        delete args['_printerIndex'];

        let params: gws.api.base.printer.Request = {
            type: gws.api.base.printer.RequestType.template,
            args,
            printerUid: prt.uid,
            dpi,
            maps: [mapParams],
        };

        await this.startJob(this.app.server.printerStart(params, {binaryRequest: false}));
    }

    async startScreenshot() {
        let dpi = Number(this.getValue('printerScreenshotDpi')) || MIN_SNAPSHOT_DPI;

        let mapParams = await this.map.printParams(
            this.previewBox.getBoundingClientRect(),
            dpi
        );

        let vs = this.map.viewState;

        let params: gws.api.base.printer.Request = {
            type: gws.api.base.printer.RequestType.map,
            maps: [mapParams],
            // format: 'png',
            dpi: dpi,
            outputSize: [
                Number(this.getValue('printerScreenshotWidth')) || DEFAULT_SNAPSHOT_SIZE,
                Number(this.getValue('printerScreenshotHeight')) || DEFAULT_SNAPSHOT_SIZE,
            ],
            // sections: [
            //     {
            //         center: [vs.centerX, vs.centerY] as gws.api.Point,
            //     }
            // ]
        };

        await this.startJob(this.app.server.printerStart(params, {binaryRequest: true}));

    }

    async startJob(res) {
        this.update({
            printerJob: {state: gws.api.core.JobState.init}
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

            case gws.api.core.JobState.init:
                this.update({printerState: 'printing'});
                break;

            case gws.api.core.JobState.open:
            case gws.api.core.JobState.running:
                this.update({printerState: 'printing'});
                this.jobTimer = setTimeout(() => this.poll(), JOB_POLL_INTERVAL);
                break;

            case gws.api.core.JobState.cancel:
                this.stop();
                break;

            case gws.api.core.JobState.complete:
                if (this.getValue('printerMode') === 'screenshot') {
                    gws.lib.downloadUrl(job.url, 'image.png', null);
                    this.stop()
                } else {
                    this.update({printerState: job.state});
                }
                break;

            case gws.api.core.JobState.error:
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
    'Toolbar.Screenshot': ScreenshotToolbarButton,
    'Tool.Print.Print': PrintTool,
    'Tool.Print.Screenshot': ScreenshotTool,
});
