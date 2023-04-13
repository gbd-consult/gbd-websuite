import * as React from 'react';


import * as gws from 'gws';
import * as components from 'gws/components';
import * as toolbar from 'gws/elements/toolbar';
import {FormField} from "gws/components/form";

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Printer';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as Controller;

const JOB_POLL_INTERVAL = 2000;
const DEFAULT_SNAPSHOT_SIZE = 300;
const MIN_SNAPSHOT_DPI = 70;
const MAX_SNAPSHOT_DPI = 600;

interface ViewProps extends gws.types.ViewProps {
    controller: Controller;
    printJob?: gws.api.base.printer.StatusResponse;
    printState?: 'preview' | 'printing' | 'error' | 'complete';
    printFormData: object;
    printSnapshotMode: boolean;
    printSnapshotDpi: number;
    printSnapshotWidth: number;
    printSnapshotHeight: number;
    printDialogZoomed: boolean;
}

const StoreKeys = [
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

class PrintToolbarButton extends toolbar.Button {
    iconClass = 'printerPrintToolbarButton';
    tool = 'Tool.Print.Print';

    get tooltip() {
        return this.__('printerPrintToolbarButton');
    }
}

class SnapshotToolbarButton extends toolbar.Button {
    iconClass = 'printerSnapshotToolbarButton';
    tool = 'Tool.Print.Snapshot';

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
        let values =         this.props.printFormData || {};


        let templateItems = this.templateItems();
        if (templateItems.length > 1) {
            fields.push({
                type: "integer",
                name: "_templateIndex",
                title: this.__('printerTemplate'),
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
                title: this.__('printerQuality'),
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
                widget={cc.createWidget(f, values)}
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
                        label={this.__('printerSnapshotResolution')}
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

        // return <overview.SmallMap controller={this.props.controller} boxSize={[w, h]}/>;
        return null;
    }

    optionsDialog() {
        let form = this.props.printSnapshotMode
            ? this.snapshotOptionsForm()
            : this.printOptionsForm();

        let ok = this.props.printSnapshotMode
            ? <gws.ui.Button
                {...gws.lib.cls('printerPreviewSnapshotButton')}
                whenTouched={() => this.props.controller.startSnapshot()}
                tooltip={this.__('printerPreviewSnapshotButton')}
            />
            : <gws.ui.Button
                {...gws.lib.cls('printerPreviewPrintButton')}
                whenTouched={() => this.props.controller.whenStartPrintButtonTouched()}
                tooltip={this.__('printerPreviewPrintButton')}
            />;

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
                            whenTouched={() => this.props.controller.app.stopTool('Tool.Print')}
                            tooltip={this.__('printerCancel')}
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
            <div className="printerPreviewBox" ref={this.boxRef} style={style}/>
            {this.props.printSnapshotMode && <div
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

class Dialog extends gws.View<ViewProps> {

    render() {
        let ps = this.props.printState;
        let job = this.props.printJob;

        let cancel = () => this.props.controller.cancelPrinting();
        let stop = () => this.props.controller.stop();

        if (ps === 'printing') {

            let label = '';

            if (job.stepType === 'layer' && job.stepName)
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
        return this.app.project.printer.templates || [];
    }

    get selectedTemplate() {
        let fd = this.getValue('printFormData') || {},
            idx = Number(fd['_templateIndex']) || 0;
        return this.templates[idx] || this.templates[0];
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

    createWidget(field: gws.types.IModelField, values: gws.types.Dict): React.ReactElement | null {
        let p = field.widgetProps;

        if (!p)
            return null;

        let tag = 'ModelWidget.' + p.type;
        let controller = (this.app.controllerByTag(tag) || this.app.createControllerFromConfig(this, {tag})) as gws.types.IModelWidget;

        let props: gws.types.Dict = {
            controller,
            field,
            values,
            whenChanged: val => this.whenWidgetChanged(field, val),
            whenEntered: val => this.whenWidgetEntered(field, val),
        }


        return controller.view(props)
    }

    whenWidgetChanged(field: gws.types.IModelField, value) {
        
    }
    whenWidgetEntered(field: gws.types.IModelField, value) {
        
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

    async whenStartPrintButtonTouched() {
        let fd = this.getValue('printFormData') || {},
            qualityLevel = Number(fd['_quality']) || 0,
            template = this.selectedTemplate,
            level = template.qualityLevels[qualityLevel],
            dpi = level ? level.dpi : 0;

        let mapParams = await this.map.printParams(
            this.previewBox.getBoundingClientRect(),
            dpi
        );

        let vs = this.map.viewState;

        let context = {...fd};
        delete context['_quality'];
        delete context['_templateIndex'];

        let params: gws.api.base.printer.Request = {
            type: gws.api.base.printer.RequestType.template,
            templateUid: template.uid,
            qualityLevel,
            maps: [mapParams],
            // sections: [
            //     {
            //         center: [vs.centerX, vs.centerY] as gws.api.Point,
            //         context,
            //     }
            // ]
        };

        await this.startJob(this.app.server.printerStart(params, {binary: false}));
    }

    async startSnapshot() {
        let dpi = Number(this.getValue('printSnapshotDpi')) || MIN_SNAPSHOT_DPI;

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
            printJob: {state: gws.api.base.printer.State.init}
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

            case gws.api.base.printer.State.init:
                this.update({printState: 'printing'});
                break;

            case gws.api.base.printer.State.open:
            case gws.api.base.printer.State.running:
                this.update({printState: 'printing'});
                this.jobTimer = setTimeout(() => this.poll(), JOB_POLL_INTERVAL);
                break;

            case gws.api.base.printer.State.cancel:
                this.stop();
                break;

            case gws.api.base.printer.State.complete:
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
            case gws.api.base.printer.State.error:
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

gws.registerTags({
    [MASTER]: Controller,
    'Toolbar.Print': PrintToolbarButton,
    'Toolbar.Snapshot': SnapshotToolbarButton,
    'Tool.Print.Print': PrintTool,
    'Tool.Print.Snapshot': SnapshotTool,
});
