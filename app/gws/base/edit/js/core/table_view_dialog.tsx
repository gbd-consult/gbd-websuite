import * as React from 'react';

import * as gws from 'gws';
import * as types from './types';
import type {Controller} from './controller';

let {Form, Row, Cell, VBox, VRow} = gws.ui.Layout;


export class TableViewDialog extends gws.View<types.ViewProps> {
    tableRef: React.RefObject<any>;

    constructor(props) {
        super(props);
        this.tableRef = React.createRef();
    }

    master() {
        return this.props.controller as Controller;
    }

    async saveFeature(feature) {
        let cc = this.master();

        cc.updateEditState({tableViewLoading: true})

        let model = cc.editState.tableViewSelectedModel;
        let fields = model.tableViewColumns.map(c => c.field);
        let ok = await cc.saveFeature(feature, fields);

        if (ok) {
            cc.featureCache.clear();
            await this.makeAndCacheRows()
        }

        cc.updateEditState({tableViewLoading: false})
        return ok;
    }

    makeAllRows(model) {
        let cc = this.master();
        let features = cc.featureCache.getForModel(model);
        let rows = []

        for (let feature of features) {
            let row = this.makeRow(feature)
            rows.push(row)
        }

        return rows;
    }


    makeRow(feature: gws.types.IFeature): types.TableViewRow {
        let cc = this.master();
        let cells = [];
        let values = feature.attributes;
        for (let col of feature.model.tableViewColumns) {
            cells.push(cc.createWidget(
                gws.types.ModelWidgetMode.cell,
                col.field,
                feature,
                values,
                this.whenWidgetChanged.bind(this),
                this.whenWidgetEntered.bind(this),
            ))
        }
        return {cells, featureUid: feature.uid}
    }

    makeSelectedRow(feature: gws.types.IFeature): types.TableViewRow {
        let cc = this.master();
        let values = feature.currentAttributes()
        let errors = cc.editState.formErrors || {}
        let cells = [];

        for (let col of feature.model.tableViewColumns) {
            let err = errors[col.field.name]
            let widget = cc.createWidget(
                gws.types.ModelWidgetMode.activeCell,
                col.field,
                feature,
                values,
                this.whenWidgetChanged.bind(this),
                this.whenWidgetEntered.bind(this),
            )
            cells.push(<React.Fragment>
                {widget}
                <div {...gws.lib.cls('editTableError', err && 'isActive')}>
                    {err || ' '}
                </div>
            </React.Fragment>)
        }
        return {cells, featureUid: feature.uid}
    }


    async makeAndCacheRows() {
        let cc = this.master();
        cc.updateEditState({tableViewLoading: true})

        let model = cc.editState.tableViewSelectedModel;
        await cc.featureCache.updateForModel(model);

        for (let field of model.fields) {
            await cc.initWidget(field);
        }

        let rows = this.makeAllRows(model)

        cc.updateEditState({tableViewRows: rows})
        cc.updateEditState({tableViewLoading: false})
    }

    closeDialog() {
        let cc = this.master();
        cc.updateEditState({
            tableViewSelectedModel: null,
            tableViewSelectedFeature: null,
            tableViewRows: null,
        })


    }

    async whenCellTouched(evt) {
        let cc = this.master();
        let es = cc.editState;
        let sf = es.tableViewSelectedFeature;

        let model = es.tableViewSelectedModel;
        let features = cc.featureCache.getForModel(model);

        let pos = [
            evt.currentTarget.parentNode.sectionRowIndex,
            evt.currentTarget.cellIndex,
            0,
        ];

        let nf = features[pos[0]];

        if (!nf || sf === nf)
            return;

        if (sf && sf.isDirty) {
            let ok = await this.saveFeature(sf);
            if (!ok)
                return
        }

        cc.updateEditState({
            tableViewSelectedFeature: nf,
            tableViewTouchPos: pos,
            serverError: '',
        })
    }

    whenWidgetChanged(feature: gws.types.IFeature, field: gws.types.IModelField, value: any) {
        let cc = this.master();
        feature.editAttribute(field.name, value);
        cc.updateEditState();
    }

    async whenWidgetEntered(feature: gws.types.IFeature, field: gws.types.IModelField, value: any) {
        let cc = this.master();
        let es = cc.editState;
        let model = es.tableViewSelectedModel;
        let features = cc.featureCache.getForModel(model);

        let rowIndex = features.indexOf(feature)
        let cellIndex = model.fields.indexOf(field)

        let ok = await this.saveFeature(feature);
        if (!ok)
            return

        let nf = features[rowIndex + 1]
        cc.updateEditState({
            tableViewSelectedFeature: nf,
            tableViewTouchPos: [rowIndex + 1, cellIndex, 0],
        })


    }


    async whenSaveButtonTouched(feature) {
        let cc = this.master();

        let ok = await this.saveFeature(feature);
        if (!ok)
            return
        cc.updateEditState({
            tableViewSelectedFeature: null,
        })


    }


    whenResetButtonTouched(feature) {
        let cc = this.master();

        feature.resetEdits();
        cc.updateEditState()
    }

    whenOpenFormButtonTouched(feature) {
        let cc = this.master();
        cc.updateEditState({
            tableViewSelectedModel: null,
            tableViewSelectedFeature: null,
            tableViewRows: null,
            sidebarSelectedFeature: feature,
        })
    }

    whenCancelButtonTouched(feature) {
        let cc = this.master();
        cc.updateEditState({
            tableViewSelectedFeature: null,
            serverError: '',
            formErrors: null,
        })
    }

    whenDeleteButtonTouched(feature) {
    }

    async whenNewButtonTouched() {
        let cc = this.master();
        let es = cc.editState;
        let feature = await cc.createFeature(cc.editState.tableViewSelectedModel);
        let model = es.tableViewSelectedModel;
        let features = cc.featureCache.getForModel(model);

        cc.updateEditState({
            tableViewSelectedFeature: feature,
            tableViewTouchPos: [features.length, 0, 1],
            serverError: '',
            formErrors: null,
        })
        await this.makeAndCacheRows()


    }


    componentDidUpdate(prevProps) {
        let cc = this.master();
        let pos = this.props.editState.tableViewTouchPos;

        if (pos && this.tableRef.current) {
            cc.updateEditState({tableViewTouchPos: null})

            let sel = `.uiTable > table > tbody > tr:nth-child(${pos[0] + 1}) > td:nth-child(${pos[1] + 1})`
            let td = this.tableRef.current.querySelector(sel)
            let el

            if (td) {
                if (pos[2])
                    td.scrollIntoView(false)
                el = td.querySelector('input')
                if (el) {
                    el.focus()
                    return
                }
                el = td.querySelector('textarea')
                if (el) {
                    el.focus()
                    return
                }
            }

        }
    }


    async componentDidMount() {
        await this.makeAndCacheRows()
    }

    dialogFooter(sf) {
        let cc = this.master();

        let pager = '';
        let error = cc.editState.serverError;
        let buttons;

        if (sf) {

            buttons = [
                <gws.ui.Button
                    {...gws.lib.cls('editSaveButton', sf.isDirty && 'isActive')}
                    tooltip={this.__('editSave')}
                    whenTouched={() => this.whenSaveButtonTouched(sf)}
                />,
                <gws.ui.Button
                    {...gws.lib.cls('editResetButton', sf.isDirty && 'isActive')}
                    tooltip={this.__('editReset')}
                    whenTouched={() => this.whenResetButtonTouched(sf)}
                />,
                <gws.ui.Button
                    {...gws.lib.cls('editOpenFormButton')}
                    tooltip={this.__('editOpenForm')}
                    whenTouched={() => this.whenOpenFormButtonTouched(sf)}
                />,
                <gws.ui.Button
                    {...gws.lib.cls('editCancelButton')}
                    tooltip={this.__('editCancel')}
                    whenTouched={() => this.whenCancelButtonTouched(sf)}
                />,
                // <gws.ui.Button
                //     {...gws.lib.cls('editDeleteButton')}
                //     tooltip={this.__('editDelete')}
                //     whenTouched={() => this.whenDeleteButtonTouched(sf)}
                // />,
            ]
        } else {
            buttons = [
                <gws.ui.Button
                    {...gws.lib.cls('editNewButton')}
                    tooltip={this.__('editNew')}
                    whenTouched={() => this.whenNewButtonTouched()}
                />,
            ]
        }

        return <Row>
            {pager}
            <Cell>
                {error && <gws.ui.Error text={error}/>}
            </Cell>
            <Cell flex/>
            {
                cc.editState.tableViewLoading
                    ? <Cell>
                        <gws.ui.Loader/>
                    </Cell>
                    : buttons.map((b, n) => <Cell key={n}>{b}</Cell>)
            }
        </Row>;

    }

    renderTable() {
        let cc = this.master();
        let model = cc.editState.tableViewSelectedModel;

        let savedRows = cc.editState.tableViewRows;

        if (gws.lib.isEmpty(savedRows)) {
            return null;
        }

        let sf = cc.editState.tableViewSelectedFeature;
        let selectedIndex = sf ? savedRows.findIndex(r => r.featureUid === sf.uid) : -1;

        let headerRow = []
        let filterRow = []

        for (let col of model.tableViewColumns) {
            headerRow.push(col.field.title)
            filterRow.push(<gws.ui.TextInput value={''} withClear={true}/>)
        }

        let rows = savedRows;

        if (sf) {
            if (selectedIndex >= 0) {
                rows = savedRows.slice(0, selectedIndex)
                rows.push(this.makeSelectedRow(sf))
                rows = rows.concat(savedRows.slice(selectedIndex + 1))
            } else {
                rows = savedRows.slice(0)
                rows.push(this.makeSelectedRow(sf))
                selectedIndex = savedRows.length

            }
        }

        let widths = [];
        for (let col of model.tableViewColumns) {
            widths.push(col.width);
        }

        return <gws.ui.Table
            tableRef={this.tableRef}
            rows={rows.map(r => r.cells)}
            selectedIndex={selectedIndex}
            fixedLeftColumn={false}
            fixedRightColumn={false}
            // headers={[headerRow, filterRow]}
            headers={[headerRow]}
            columnWidths={widths}
            whenCellTouched={e => this.whenCellTouched(e)}
        />
    }

    render() {
        let cc = this.master();
        let model = cc.editState.tableViewSelectedModel;
        let sf = cc.editState.tableViewSelectedFeature;

        return <gws.ui.Dialog
            {...gws.lib.cls('editTableViewDialog', this.props.editTableViewDialogZoomed && 'isZoomed')}
            title={model.title}
            whenClosed={() => this.closeDialog()}
            // whenZoomed={() => this.props.controller.update({editTableViewDialogZoomed: !this.props.editTableViewDialogZoomed})}
            footer={this.dialogFooter(sf)}
        >
            {this.renderTable()}
        </gws.ui.Dialog>;
    }
}
