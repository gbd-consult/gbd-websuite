import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './common/toolbar';

let {Form, Row, Cell} = gws.ui.Layout;

interface GeorisksViewProps extends gws.types.ViewProps {
    controller: GeorisksController;
    georisksX: string;
    georisksY: string;
    georisksDialogMode: string;
    georisksFormName: string;
    georisksFormMessage: string;
    georisksFormFiles: FileList;
    georisksFormPrivacyAccept: boolean;
}

const GeorisksStoreKeys = [
    'georisksDialogMode',
    'georisksX',
    'georisksY',
    'georisksFormName',
    'georisksFormMessage',
    'georisksFormFiles',
    'georisksFormPrivacyAccept',
];

class GeorisksDialog extends gws.View<GeorisksViewProps> {

    close() {
        this.props.controller.update({georisksDialogMode: ''});
    }

    form() {
        let data = [
            {
                name: 'georisksFormName',
                title: this.__('modGeorisksReportFormName'),
                value: this.props.georisksFormName,
                editable: true,
            },
            {
                name: 'georisksFormMessage',
                title: this.__('modGeorisksReportFormMessage'),
                type: 'text',
                value: this.props.georisksFormMessage,
                editable: true,
            },
            {
                name: 'georisksFormFiles',
                title: this.__('modGeorisksReportFormFiles'),
                type: 'file',
                value: this.props.georisksFormFiles,
                editable: true,
                accept: 'image/jpeg',
                mutlitple: true,
            },
        ];

        let privacyText = () => {
            // must be like "I have read $the policy$ and accept it"
            let t = this.__('modGeorisksReportPrivacyLink').split('$');
            let h = this.app.actions['georisks']['privacyPolicyLink'];
            return <Cell flex>
                {t[0]}
                <gws.ui.Link href={h} target="_blank" content={t[1]}/>
                {t[2]}
            </Cell>;

        };

        return <Form>
            <Row>
                <Cell flex>
                    <gws.components.sheet.Editor
                        data={data}
                        whenChanged={(k, v) => this.props.controller.update({[k]: v})}
                    />
                </Cell>
            </Row>
            <Row>
                <Cell>
                    <gws.ui.Toggle
                        type="checkbox"
                        value={this.props.georisksFormPrivacyAccept}
                        whenChanged={v => this.props.controller.update({georisksFormPrivacyAccept: v})}
                    />
                </Cell>
                {privacyText()}
            </Row>
            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.IconButton
                        className="cmpButtonFormOk"
                        disabled={!this.props.controller.reportFormIsValid()}
                        whenTouched={() => this.props.controller.submitReportForm()}
                    />
                </Cell>
                <Cell>
                    <gws.ui.IconButton
                        className="cmpButtonFormCancel"
                        whenTouched={() => this.close()}
                    />
                </Cell>
            </Row>
        </Form>
    }

    message(mode) {
        switch (mode) {
            case 'loading':
                return <div className="modGeorisksFormPadding">
                    <p>{this.__('modGeorisksDialogLoading')}</p>
                    <gws.ui.Loader/>
                </div>;

            case 'success':
                return <div className="modGeorisksFormPadding">
                    <p>{this.__('modGeorisksDialogSuccess')}</p>
                </div>;

            case 'error':
                return <div className="modGeorisksFormPadding">
                    <gws.ui.Error text={this.__('modGeorisksDialogError')}/>
                </div>;
        }

    }

    render() {
        let mode = this.props.georisksDialogMode;
        if (!mode)
            return null;

        if (mode === 'open') {
            return <gws.ui.Dialog
                className="modGeorisksDialog"
                title={this.__('modGeorisksReportDialogTitle')}
                whenClosed={() => this.close()}
            >{this.form()}</gws.ui.Dialog>
        }

        if (mode === 'loading') {
            return <gws.ui.Dialog
                className='modGeorisksSmallDialog'
            >{this.message(mode)}</gws.ui.Dialog>
        }

        return <gws.ui.Dialog
            className='modGeorisksSmallDialog'
            whenClosed={() => this.close()}
        >{this.message(mode)}</gws.ui.Dialog>

    }
}

class GeorisksClickTool extends gws.Tool {

    async run(evt) {
        this.update({
            georisksX: evt.coordinate[0],
            georisksY: evt.coordinate[1],
            georisksDialogMode: 'open',
        })
    }

    start() {

        this.map.prependInteractions([
            this.map.pointerInteraction({
                whenTouched: evt => this.run(evt),
            }),
        ]);
    }

    stop() {
    }

}

interface ValidationRule {
    type: string;
    minLen?: number;
    maxLen?: number;

}

function isValid(cc: gws.Controller, value: any, rule: ValidationRule): boolean {
    let t = rule['type'];

    let len = (v) => {
        if (rule.minLen && v.length < rule.minLen)
            return false;
        if (rule.maxLen && v.length > rule.maxLen)
            return false;
        return true;
    };

    if (t === 'string') {
        return len((value ? String(value) : '').trim());
    }

    if (t === 'true') {
        return !!value;
    }

    if (t === 'fileList') {
        return len(value || []);
    }
}

function validateStoreValues(cc: gws.Controller, rules: { [k: string]: ValidationRule }) {
    return Object.keys(rules).every(k => isValid(cc, cc.getValue(k), rules[k]));
}

class GeorisksToolbarButton extends toolbar.Button {
    canInit() {
        return !!this.app.actions['georisks'];
    }

    iconClass = 'modGeorisksToolbarButton';
    tool = 'Tool.Georisks.Click';

    get tooltip() {
        return this.__('modGeorisksToolbarButton');
    }
}

class GeorisksController extends gws.Controller {
    canInit() {
        return !!this.app.actions['georisks'];
    }

    async init() {
        // this.app.whenLoaded(() => {
        //     this.app.startTool('Tool.Georisks.Click');
        // })
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(GeorisksDialog, GeorisksStoreKeys));
    }

    reportFormIsValid() {
        let rules = {
            georisksFormName: {type: 'string', minLen: 1},
            georisksFormMessage: {type: 'string', minLen: 1},
            georisksFormFiles: {type: 'fileList', minLen: 1, maxLen: 1, maxTotalSize: 1e6},
            georisksFormPrivacyAccept: {type: 'true'}
        };

        return validateStoreValues(this, rules);
    }

    async submitReportForm() {
        this.update({georisksDialogMode: 'loading'});

        let readFile = (file: File) => new Promise<gws.api.GeorisksReportFile>((resolve, reject) => {

            let reader = new FileReader();

            reader.onload = function () {
                let b = reader.result as ArrayBuffer;
                resolve({
                    content: new Uint8Array(b)
                });
            };

            reader.onabort = reject;
            reader.onerror = reject;

            reader.readAsArrayBuffer(file);

        });

        let files: Array<gws.api.GeorisksReportFile> = [],
            fileList: FileList = this.getValue('georisksFormFiles');

        if (fileList && fileList.length) {
            let fs: Array<File> = [].slice.call(fileList, 0);
            files = await Promise.all(fs.map(readFile));
        }

        let params: gws.api.GeorisksCreateReportParams = {
            shape: this.map.geom2shape(new ol.geom.Point([
                this.getValue('georisksX'),
                this.getValue('georisksY')
            ])),
            name: this.getValue('georisksFormName'),
            message: this.getValue('georisksFormMessage'),
            projectUid: this.app.project.uid,
            files
        };

        let res = await this.app.server.georisksCreateReport(params, {binary: true});

        this.update({georisksDialogMode: res.error ? 'error' : 'success'});
    }

}

export const tags = {
    'Shared.Georisks': GeorisksController,
    'Toolbar.Georisks': GeorisksToolbarButton,
    'Tool.Georisks.Click': GeorisksClickTool,
};
