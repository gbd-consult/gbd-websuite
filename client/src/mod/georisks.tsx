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

    georisksFormCategory: string;
    georisksFormVolume: string;
    georisksFormHeight: string;

    georisksFormKind_hangrutschung: string;
    georisksFormKind_mure: string;

    georisksFormDanger_street: boolean,
    georisksFormDanger_rail: boolean,
    georisksFormDanger_way: boolean,
    georisksFormDanger_house: boolean,
    georisksFormDanger_object: boolean,
    georisksFormDanger_person: boolean,

    georisksFormMessage: string;
    georisksFormFiles: FileList;

    georisksFormPrivacyAccept: boolean;

}

const GeorisksStoreKeys = [
    'georisksDialogMode',
    'georisksX',
    'georisksY',

    'georisksFormCategory',
    'georisksFormVolume',
    'georisksFormHeight',

    'georisksFormKind_hangrutschung',
    'georisksFormKind_mure',

    'georisksFormDanger_street',
    'georisksFormDanger_rail',
    'georisksFormDanger_way',
    'georisksFormDanger_house',
    'georisksFormDanger_object',
    'georisksFormDanger_person',

    'georisksFormMessage',
    'georisksFormFiles',

    'georisksFormPrivacyAccept',
];

const MAX_FILES = 5;

const validationRules = [
    {category: /./, key: 'georisksFormCategory', type: 'string', minLen: 1},
    {category: /./, key: 'georisksFormMessage', type: 'string', minLen: 1},
    {category: /./, key: 'georisksFormFiles', type: 'fileList', minLen: 1, maxLen: MAX_FILES, maxTotalSize: 1e6},
    {category: /./, key: 'georisksFormPrivacyAccept', type: 'true'},
    {category: /blockschlag|grossblockschlag|felssturz/, key: 'georisksFormVolume', type: 'string', minLen: 1},
    {category: /hangrutschung/, key: 'georisksFormKind_hangrutschung', type: 'string', minLen: 1},
    {category: /mure/, key: 'georisksFormKind_mure', type: 'string', minLen: 1},
];

const DANGERS = ['street', 'rail', 'way', 'house', 'object', 'person'];

class GeorisksForm extends gws.View<GeorisksViewProps> {
    render() {
        let menu = {

            category: [
                {
                    value: 'steinschlag',
                    text: this.__('modGeorisksReportFormCat_steinschlag')
                },
                {
                    value: 'blockschlag',
                    text: this.__('modGeorisksReportFormCat_blockschlag')
                },
                {
                    value: 'grossblockschlag',
                    text: this.__('modGeorisksReportFormCat_grossblockschlag')
                },
                {
                    value: 'felssturz',
                    text: this.__('modGeorisksReportFormCat_felssturz')
                },
                {
                    value: 'hangrutschung',
                    text: this.__('modGeorisksReportFormCat_hangrutschung')
                },
                {
                    value: 'mure',
                    text: this.__('modGeorisksReportFormCat_mure')
                },
            ],

            volume_blockschlag: [
                {
                    value: '1-5',
                    text: this.__('modGeorisksReportFormVol_1'),
                },
                {
                    value: '5-10',
                    text: this.__('modGeorisksReportFormVol_5'),
                },
            ],

            volume_grossblockschlag: [
                {
                    value: '1-5',
                    text: this.__('modGeorisksReportFormVol_1'),
                },
                {
                    value: '5-10',
                    text: this.__('modGeorisksReportFormVol_5'),
                },
            ],

            volume_felssturz: [
                {
                    text: this.__('modGeorisksReportFormVol_10'),
                    value: '10-100',
                },
                {
                    text: this.__('modGeorisksReportFormVol_100'),
                    value: '100-1000',
                },
                {
                    text: this.__('modGeorisksReportFormVol_1000'),
                    value: '1000-10000',
                },
                {
                    text: this.__('modGeorisksReportFormVol_10000'),
                    value: '10000-',
                },
            ],

            kind_hangrutschung: [

                {
                    text: this.__('modGeorisksReportFormKind_allgemein'),
                    value: 'allgemein',
                },
                {
                    text: this.__('modGeorisksReportFormKind_tief'),
                    value: 'tief',
                },
                {
                    text: this.__('modGeorisksReportFormKind_flach'),
                    value: 'flach',
                },
            ],

            kind_mure: [

                {
                    text: this.__('modGeorisksReportFormKind_schutt'),
                    value: 'schutt',
                },
                {
                    text: this.__('modGeorisksReportFormKind_block'),
                    value: 'block',
                },
                {
                    text: this.__('modGeorisksReportFormKind_murgang'),
                    value: 'murgang',
                },
            ],

            height: [
                {
                    text: '5',
                    value: '5',
                },
                {
                    text: '10',
                    value: '10',
                },
                {
                    text: '15',
                    value: '15',
                },
            ],

        }

        let cat = this.props.georisksFormCategory || '';

        let select = (prop, items, combo = false) => <gws.ui.Select
            items={items}
            value={this.props[prop]}
            withCombo={combo}
            whenChanged={v => whenChanged(prop, v)}
        />;

        let whenChanged = (name, v) => {
            this.props.controller.update({[name]: v});
        }

        return <table className="cmpPropertySheet">
            <tbody>
            <tr>
                <th>{this.__('modGeorisksReportFormLabelEvent')}</th>
                <td colSpan={2}>{select('georisksFormCategory', menu.category)}</td>
            </tr>
            {
                cat === 'blockschlag' &&
                <tr>
                    <th>{this.__('modGeorisksReportFormLabelVolume')}</th>
                    <td>{select('georisksFormVolume', menu.volume_blockschlag)}</td>
                    <td><b>m³</b></td>
                </tr>

            }
            {
                cat === 'grossblockschlag' &&
                <tr>
                    <th>{this.__('modGeorisksReportFormLabelVolume')}</th>
                    <td>{select('georisksFormVolume', menu.volume_grossblockschlag)}</td>
                    <td><b>m³</b></td>
                </tr>
            }
            {
                cat === 'felssturz' &&
                <tr>
                    <th>{this.__('modGeorisksReportFormLabelVolume')}</th>
                    <td>{select('georisksFormVolume', menu.volume_felssturz)}</td>
                    <td><b>m³</b></td>
                </tr>
            }
            {
                cat.match(/^(steinschlag|blockschlag|grossblockschlag|felssturz)$/) &&
                <tr>
                    <th>{this.__('modGeorisksReportFormLabelHeight')}</th>
                    <td>{select('georisksFormHeight', menu.height, true)}</td>
                    <td><b>m</b></td>
                </tr>
            }
            {
                cat === 'hangrutschung' &&
                <tr>
                    <th>{this.__('modGeorisksReportFormLabelKind')}</th>
                    <td colSpan={2}>{select('georisksFormKind_hangrutschung', menu.kind_hangrutschung)}</td>
                </tr>
            }
            {
                cat === 'mure' &&
                <tr>
                    <th>{this.__('modGeorisksReportFormLabelKind')}</th>
                    <td colSpan={2}>{select('georisksFormKind_mure', menu.kind_mure)}</td>
                </tr>
            }
            <tr>
                <th>{this.__('modGeorisksReportFormLabelDanger')}</th>
                <td colSpan={2}>
                    {DANGERS.map(d => <gws.ui.Toggle
                        type="checkbox"
                        label={this.__('modGeorisksReportFormDanger_' + d)}
                        value={this.props['georisksFormDanger_' + d]}
                        whenChanged={v => whenChanged('georisksFormDanger_' + d, v)}
                    />)}
                </td>
            </tr>
            <tr>
                <th>{this.__('modGeorisksReportFormLabelMessage')}</th>
                <td colSpan={2}>
                    <gws.ui.TextArea
                        height={100}
                        value={this.props.georisksFormMessage}
                        whenChanged={v => whenChanged('georisksFormMessage', v)}
                    />
                </td>
            </tr>
            <tr>
                <th>{this.__('modGeorisksReportFormLabelFiles')}</th>
                <td colSpan={2}>
                    <gws.ui.FileInput
                        accept="image/jpeg"
                        multiple={true}
                        value={this.props.georisksFormFiles}
                        whenChanged={v => whenChanged('georisksFormFiles', v)}
                    />
                </td>
            </tr>

            </tbody>
        </table>
    }
}

class GeorisksDialog extends gws.View<GeorisksViewProps> {

    close() {
        this.props.controller.update({georisksDialogMode: ''});
    }

    form() {

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
                <GeorisksForm {...this.props} />
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
    key: string;
    type: string;
    minLen?: number;
    maxLen?: number;

}

function isValid(cc: gws.Controller, rule: ValidationRule): boolean {
    let value = cc.getValue(rule.key);
    let t = rule.type;

    let checkLength = (v) => {
        if (rule.minLen && v.length < rule.minLen)
            return false;
        if (rule.maxLen && v.length > rule.maxLen)
            return false;
        return true;
    };

    if (t === 'string') {
        return checkLength((value ? String(value) : '').trim());
    }

    if (t === 'true') {
        return !!value;
    }

    if (t === 'fileList') {
        return checkLength(value || []);
    }
}

function validateStoreValues(cc: gws.Controller, rules: Array<ValidationRule>) {
    return rules.every(rule => isValid(cc, rule));
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
        //     this.update({
        //         georisksDialogMode: 'open',
        //     })
        // })

    }

    get appOverlayView() {
        return this.createElement(
            this.connect(GeorisksDialog, GeorisksStoreKeys));
    }

    reportFormIsValid() {
        let cat = this.getValue('georisksFormCategory');

        if (!cat)
            return false;

        let rules = validationRules.filter(rule => rule.category.test(cat));
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

        let cat = this.getValue('georisksFormCategory');

        let params: gws.api.GeorisksCreateReportParams = {
            shape: this.map.geom2shape(new ol.geom.Point([
                this.getValue('georisksX'),
                this.getValue('georisksY')
            ])),

            category: this.getValue('georisksFormCategory'),
            volume: this.getValue('georisksFormVolume'),
            height: this.getValue('georisksFormHeight'),
            kind: this.getValue('georisksFormKind_' + cat) || '',
            message: this.getValue('georisksFormMessage'),
            dangers: DANGERS.filter(d => this.getValue('georisksFormDanger_' + d)).join(),
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
