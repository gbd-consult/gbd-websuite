import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './toolbar';

let {Form, Row, Cell} = gws.ui.Layout;

interface GeorisksViewProps extends gws.types.ViewProps {
    controller: GeorisksController;
    georisksX: string;
    georisksY: string;
    georisksDialogMode: string;

    georisksFormCategory: string;
    georisksFormVolume: string;
    georisksFormHeight: string;

    georisksFormKind_mudslide: string;

    georisksFormDanger_road: boolean;
    georisksFormDanger_railway: boolean;
    georisksFormDanger_path: boolean;
    georisksFormDanger_buildings: boolean;
    georisksFormDanger_pasture: boolean;
    georisksFormDanger_arable: boolean;
    georisksFormDanger_infrastructure: boolean;
    georisksFormDanger_casualties: boolean;

    georisksFormMessage: string;
    georisksFormDate: string;
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

    'georisksFormKind_mudslide',

    'georisksFormDanger_road',
    'georisksFormDanger_railway',
    'georisksFormDanger_path',
    'georisksFormDanger_buildings',
    'georisksFormDanger_pasture',
    'georisksFormDanger_arable',
    'georisksFormDanger_infrastructure',
    'georisksFormDanger_casualties',

    'georisksFormMessage',
    'georisksFormDate',
    'georisksFormFiles',

    'georisksFormPrivacyAccept',
];

const MAX_FILES = 5;

const validationRules = [
    {category: /./, key: 'georisksFormCategory', type: 'string', minLen: 1},
    {category: /./, key: 'georisksFormMessage', type: 'string', minLen: 1},
    {category: /./, key: 'georisksFormDate', type: 'string', minLen: 1},
    {category: /./, key: 'georisksFormFiles', type: 'fileList', minLen: 0, maxLen: MAX_FILES, maxTotalSize: 1e6},
    {category: /./, key: 'georisksFormPrivacyAccept', type: 'true'},
    {category: /mudslide/, key: 'georisksFormKind_mudslide', type: 'string', minLen: 1},
];

const DANGERS = ['road', 'railway', 'path', 'buildings', 'pasture', 'arable', 'infrastructure', 'casualties'];

function formFields(cc) {

    let height = () => [
        {value: '5', text: '5'},
        {value: '10', text: '10'},
        {value: '15', text: '15'},
        {value: '20', text: '20'},
        {value: '25', text: '25'},
        {value: '30', text: '30'},
        {value: '40', text: '40'},
        {value: '40+', text: '> 40'},
    ];

    return [
        {
            value: 'rockfall',
            text: cc.__('modGeorisksReportFormCat_rockfall'),
            fields: [],
        },
        {
            value: 'mudslide',
            text: cc.__('modGeorisksReportFormCat_mudslide'),
            fields: [
                {
                    type: 'kind', prop: 'georisksFormKind_mudslide', values: [
                        {value: 'debris', text: cc.__('modGeorisksReportFormKind_debris')},
                        {value: 'mud', text: cc.__('modGeorisksReportFormKind_mud')},
                    ]
                },
            ]
        },
        {
            value: 'avalanche',
            text: cc.__('modGeorisksReportFormCat_avalanche'),
            fields: [],
        },
    ]
}

const STAR = <span className="uiRequiredStar">*</span>;

class GeorisksForm extends gws.View<GeorisksViewProps> {
    render() {

        let cat = this.props.georisksFormCategory || '';

        let allFields = formFields(this),
            catFields = allFields.find(f => f.value === cat);

        let select = (prop, items, combo = false) => <gws.ui.Select
            items={items}
            value={this.props[prop] || ''}
            withCombo={combo}
            whenChanged={v => whenChanged(prop, v)}
        />;

        let whenChanged = (name, v) => {
            this.props.controller.update({[name]: v});
        };

        let rowForField = (key, cat, cf) => {
            if (cf.type === 'volume') {
                return <tr key={key}>
                    <th>{this.__('modGeorisksReportFormLabelVolume')}{STAR}</th>
                    <td>{select(cf.prop, cf.values)}</td>
                    <td><b>m³</b></td>
                </tr>
            }

            if (cf.type === 'kind') {
                return <tr key={key}>
                    <th>{this.__('modGeorisksReportFormLabelKind')}{STAR}</th>
                    <td>{select(cf.prop, cf.values)}</td>
                </tr>
            }

            if (cf.type === 'height') {
                return <tr key={key}>
                    <th>{this.__('modGeorisksReportFormLabelHeight')}{STAR}</th>
                    <td>{select(cf.prop, cf.values, true)}</td>
                    <td><b>m</b></td>
                </tr>
            }
        };

        return <table className="cmpPropertySheet">
            <tbody>
            <tr>
                <th>{this.__('modGeorisksReportFormLabelEvent')}{STAR}</th>
                <td colSpan={2}>{select('georisksFormCategory', allFields)}</td>
            </tr>

            {catFields && catFields.fields.map((cf, n) => rowForField(n, cat, cf))}

            <tr>
                <th>{this.__('modGeorisksReportFormLabelDanger')}{STAR}</th>
                <td colSpan={2}>
                    {Object.keys(gws.api.GeorisksReportDanger).map((d, n) => <gws.ui.Toggle
                        key={n}
                        type="checkbox"
                        label={this.__('modGeorisksReportFormDanger_' + d)}
                        value={this.props['georisksFormDanger_' + d]}
                        inline={true}
                        whenChanged={v => whenChanged('georisksFormDanger_' + d, v)}
                    />)}
                </td>
            </tr>
            <tr>
                <th>{this.__('modGeorisksReportFormLabelMessage')}{STAR}</th>
                <td colSpan={2}>
                    <gws.ui.TextArea
                        height={100}
                        value={this.props.georisksFormMessage}
                        whenChanged={v => whenChanged('georisksFormMessage', v)}
                    />
                </td>
            </tr>
            <tr>
                <th>{this.__('modGeorisksReportFormLabelDate')}{STAR}</th>
                <td colSpan={2}>
                    <gws.ui.DateInput
                        value={this.props.georisksFormDate}
                        whenChanged={v => whenChanged('georisksFormDate', v)}
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
            let setup = this.app.actionSetup('georisks');

            if (!setup)
                return '';

            // must be like "I have read $the policy$ and accept it"
            let t = this.__('modGeorisksReportPrivacyLink').split('$');
            let h = setup['privacyPolicyLink'][this.app.locale.id];
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
                    <gws.ui.Button
                        className="cmpButtonFormOk"
                        disabled={!this.props.controller.reportFormIsValid()}
                        whenTouched={() => this.props.controller.submitReportForm()}
                    />
                </Cell>
                <Cell>
                    <gws.ui.Button
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
        return !!this.app.actionSetup('georisks');
    }

    iconClass = 'modGeorisksToolbarButton';
    tool = 'Tool.Georisks.Click';

    get tooltip() {
        return this.__('modGeorisksToolbarButton');
    }
}

class GeorisksController extends gws.Controller {
    canInit() {
        return !!this.app.actionSetup('georisks');
    }

    async init() {
        this.app.whenLoaded(() => {
            this.update({
                georisksFormDate: new Date().toISOString().slice(0, 10)
            })
        })

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
            // volume: this.getValue('georisksFormVolume'),
            // height: this.getValue('georisksFormHeight'),
            kind: this.getValue('georisksFormKind_' + cat) || null,
            message: this.getValue('georisksFormMessage'),
            date: this.getValue('georisksFormDate'),
            dangers:
                Object.keys(gws.api.GeorisksReportDanger)
                    .filter(d => this.getValue('georisksFormDanger_' + d))
                    .map(d => gws.api.GeorisksReportDanger[d]),
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
