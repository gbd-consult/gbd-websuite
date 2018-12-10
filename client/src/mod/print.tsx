import * as React from 'react';

import * as gws from 'gws';
import * as toolbar from './toolbar';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Printer';
const JOB_POLL_INTERVAL = 2000;

interface ViewProps extends gws.types.ViewProps {
    printJob?: gws.api.PrinterResponse;
    printQuality: string;
    printState?: 'preview' | 'printing' | 'error' | 'complete';
    printTemplateIndex: string;
    printData: object;
}

let allProps = [
    'printJob',
    'printQuality',
    'printState',
    'printTemplateIndex',
    'printData',
];

interface GoButtonProps extends ViewProps {
    controller: GoButton;
    toolbarItem: gws.types.IController;
}

class GoButtonView extends gws.View<GoButtonProps> {
    render() {
        let active = this.props.toolbarItem === this.props.controller;
        return <gws.ui.IconButton
            {...gws.tools.cls('modPrintButton', active && 'isActive')}
            tooltip={this.__('modPrintButton')}
            whenTouched={() => this.props.controller.touched()}
        />;
    }
}

class GoButton extends gws.Controller {
    isToolbarButton = true;
    parent: toolbar.Group;

    touched() {
        let master = this.app.controller(MASTER) as PrinterController;
        this.update({
            toolbarGroup: this.parent,
            toolbarItem: this,
        });
        master.startPreview()
    }

    get defaultView() {
        return this.createElement(
            this.connect(GoButtonView, [...allProps, 'toolbarItem']));
    }
}

interface PreviewBoxProps extends ViewProps {
    controller: PrinterController;
}

class PreviewBox extends gws.View<PreviewBoxProps> {

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

    dataSheet() {
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
            tpl.dataModel.forEach(attr => data.push({
                name: attr.name,
                title: attr.title || attr.name,
                value: pd[attr.name] || '',
                editable: true,
                type: attr.type,
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

    optionsDialog() {
        let vs = this.props.controller.map.viewState;

        return <div className="modPrintPreviewDialog">
            <Form>
                <Row>
                    <Cell flex>{this.dataSheet()}</Cell>
                </Row>
                <Row>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.IconButton
                            {...gws.tools.cls('modPrintPreviewPrintButton')}
                            whenTouched={() => this.props.controller.startPrinting()}
                            tooltip={this.__('modPrintButton')}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton
                            {...gws.tools.cls('modPrintPreviewCancelButton')}
                            whenTouched={() => this.props.controller.reset()}
                            tooltip={this.__('modPrintCancel')}
                        />
                    </Cell>
                </Row>
                {vs.rotation !== 0 && <gws.ui.Error text={this.__('modPrintRotationWarning')}/>}
            </Form>
        </div>
    }

    render() {
        let ps = this.props.printState;
        let tpl = this.props.controller.selectedTemplate;

        if (ps !== 'preview' || !tpl)
            return null;

        let w = gws.tools.mm2px(tpl.mapWidth);
        let h = gws.tools.mm2px(tpl.mapHeight);

        let style = {
            width: w,
            height: h,
            marginLeft: -(w >> 1),
            marginTop: -(h >> 1)
        };

        return <div>
            <div className="modPrintPreviewBox" ref={this.boxRef} style={style}/>
            {this.optionsDialog()}
        </div>;

    }

}

interface PrintDialogProps extends ViewProps {
    controller: PrinterController;
}

class PrintDialog extends gws.View<PrintDialogProps> {

    label(job) {
        let s = this.__('modPrintPrinting');

        if (job.otype === 'layer' && job.oname)
            return s + ' ' + gws.tools.shorten(job.oname, 40);

        return s + '...'

    }

    render() {
        let ps = this.props.printState;
        let job = this.props.printJob;

        if (ps === 'printing') {

            return <gws.ui.Dialog className="modPrintProgressDialog">
                <gws.ui.Progress
                    label={this.label(job)}
                    value={job.progress || 0}
                />
                <Row>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.TextButton
                            whenTouched={() => this.props.controller.cancelPrinting()}
                        >{this.__('modPrintCancel')}</gws.ui.TextButton>
                    </Cell>
                </Row>
            </gws.ui.Dialog>;
        }

        let reset = () => this.props.controller.reset();

        if (ps === 'complete') {
            return <gws.ui.Dialog className="modPrintResultDialog" whenClosed={reset}>
                <iframe src={job.url}/>
            </gws.ui.Dialog>;
        }

        if (ps === 'error') {
            return <gws.ui.Dialog className="modPrintProgressDialog" whenClosed={reset}>
                <gws.ui.Error
                    text={this.__('modPrintError')}
                    longText={this.__('modPrintErrorDetails')}

                />
            </gws.ui.Dialog>;
        }

        return null;
    }
}

class PrinterController extends gws.Controller {
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
            this.connect(PreviewBox, allProps));
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(PrintDialog, allProps));
    }

    get activeJob() {
        return this.getValue('printJob');
    }

    async init() {
        await super.init();
        this.whenChanged('printJob', job => this.jobUpdated(job));
    }

    reset() {
        this.update({
            printJob: null,
            printState: null,
            toolbarGroup: null,
            toolbarItem: null,
        });
        clearTimeout(this.jobTimer);
        this.jobTimer = 0;
    }

    startPreview() {
        return this.update({
            printState: 'preview'
        });
    }

    async startPrinting() {
        let params = await this.map.printParams(
            this.previewBox.getBoundingClientRect(),
            this.selectedTemplate,
            Number(this.getValue('printQuality')) || 0
        );

        let vs = this.map.viewState;

        params.sections = [
            {
                center: [vs.centerX, vs.centerY] as gws.api.Point,
                data: this.getValue('printData') || {}
            }
        ];

        this.update({
            printJob: {state: gws.api.JobState.init}
        });

        console.time('call_print');
        let job = await this.app.server.printerStart(params);
        console.timeEnd('call_print');

        // the job can be canceled in the mean time

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

        this.reset();
        console.log('PRINT CANCEL');

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
                this.reset();
                break;

            case gws.api.JobState.complete:
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
    [MASTER]: PrinterController,
    'Toolbar.Print.Go': GoButton,
};
