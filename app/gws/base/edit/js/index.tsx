import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from 'gws/elements/sidebar';
import * as toolbar from 'gws/elements/toolbar';
import * as draw from 'gws/elements/draw';
import * as components from 'gws/components';

// see app/gws/base/model/validator.py
const DEFAULT_MESSAGE_PREFIX = 'validationError_';

const SEARCH_TIMEOUT = 500;
const GEOMETRY_TIMEOUT = 2000;

let {Form, Row, Cell, VBox, VRow} = gws.ui.Layout;

const MASTER = 'Shared.Edit';

function _master(obj: any): Controller {
    if (obj.app)
        return obj.app.controller(MASTER) as Controller;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as Controller;
}

interface TableViewRow {
    cells: Array<React.ReactElement>;
    featureUid: string;
}


export interface EditState {
    sidebarSelectedModel?: gws.types.IModel;
    sidebarSelectedFeature?: gws.types.IFeature;
    tableViewSelectedModel?: gws.types.IModel;
    tableViewSelectedFeature?: gws.types.IFeature;
    tableViewRows: Array<TableViewRow>;
    tableViewTouchPos?: Array<number>;
    tableViewLoading: boolean;
    formErrors?: object;
    serverError?: string;
    drawModel?: gws.types.IModel;
    drawFeature?: gws.types.IFeature;
    featureHistory: Array<gws.types.IFeature>;
    featureListSearchText: { [modelUid: string]: string };
    featureCache: { [key: string]: Array<gws.types.IFeature> },
    isWaiting: boolean,
}

interface SelectModelDialogData {
    type: 'SelectModel';
    models: Array<gws.types.IModel>;
    whenSelected: (model: gws.types.IModel) => void;
}

interface SelectFeatureDialogData {
    type: 'SelectFeature';
    model: gws.types.IModel;
    field: gws.types.IModelField;
    whenFeatureTouched: (f: gws.types.IFeature) => void;
}

interface DeleteFeatureDialogData {
    type: 'DeleteFeature';
    feature: gws.types.IFeature;
    whenConfirmed: () => void;
}

interface ErrorDialogData {
    type: 'Error';
    errorText: string;
    errorDetails: string;
}

interface GeometryTextDialogData {
    type: 'GeometryText';
    shape: gws.api.base.shape.Props;
    whenSaved: (shape: gws.api.base.shape.Props) => void;
}

type DialogData =
    SelectModelDialogData
    | SelectFeatureDialogData
    | DeleteFeatureDialogData
    | ErrorDialogData
    | GeometryTextDialogData
    ;


interface ViewProps extends gws.types.ViewProps {
    controller: Controller;
    editState: EditState;
    editDialogData?: DialogData;
    editTableViewDialogZoomed: boolean;
    appActiveTool: string;
}

export const StoreKeys = [
    'editState',
    'editDialogData',
    'editTableViewDialogZoomed',
    'appActiveTool',
];

const ENABLED_SHAPES_BY_TYPE = {
    'GEOMETRY': null,
    'POINT': ['Point'],
    'LINESTRING': ['Line'],
    'POLYGON': ['Polygon', 'Circle', 'Box'],
    'MULTIPOINT': ['Point'],
    'MULTILINESTRING': ['Line'],
    'MULTIPOLYGON': ['Polygon', 'Circle', 'Box'],
    'GEOMETRYCOLLECTION': null,
};

export class PointerTool extends gws.Tool {
    oFeatureCollection: ol.Collection<ol.Feature>;
    snap: boolean = true;

    async init() {
        await super.init();
        this.app.whenChanged('mapFocusedFeature', () => {
            this.setFeature(this.getValue('mapFocusedFeature'));
        })
    }

    setFeature(feature?: gws.types.IFeature) {
        if (!this.oFeatureCollection)
            this.oFeatureCollection = new ol.Collection<ol.Feature>();

        this.oFeatureCollection.clear();

        if (!feature || !feature.geometry) {
            return;
        }

        let oFeature = feature.oFeature
        if (oFeature === this.oFeatureCollection.item(0)) {
            return;
        }

        this.oFeatureCollection.push(oFeature)
    }

    async whenPointerDown(evt: ol.MapBrowserEvent) {
        let cc = _master(this);

        if (!this.oFeatureCollection)
            this.oFeatureCollection = new ol.Collection<ol.Feature>();

        if (evt.type !== 'singleclick')
            return

        let currentFeatureClicked = false;

        cc.map.oMap.forEachFeatureAtPixel(evt.pixel, oFeature => {
            if (oFeature === this.oFeatureCollection.item(0)) {
                currentFeatureClicked = true;
            }
        });

        if (currentFeatureClicked) {
            return
        }

        await cc.whenPointerDownAtCoordinate(evt.coordinate);
    }

    start() {
        let cc = _master(this);

        this.setFeature(cc.editState.sidebarSelectedFeature);

        let ixPointer = new ol.interaction.Pointer({
            handleEvent: evt => this.whenPointerDown(evt)
        });

        let ixModify = this.map.modifyInteraction({
            features: this.oFeatureCollection,
            whenEnded: oFeatures => {
                if (oFeatures[0]) {
                    let feature = oFeatures[0]['_gwsFeature'];
                    if (feature) {
                        cc.whenModifyEnded(feature)


                    }
                }
            }
        });

        let ixs: Array<ol.interaction.Interaction> = [ixPointer, ixModify];

        // if (this.snap && cc.activeLayer)
        //     ixs.push(this.map.snapInteraction({
        //             layer: cc.activeLayer
        //         })
        //     );

        this.map.appendInteractions(ixs);

    }


    stop() {
        this.oFeatureCollection = null;
    }

}

class DrawTool extends draw.Tool {
    whenEnded(shapeType, oFeature) {
        _master(this).whenDrawEnded(oFeature);
    }

    enabledShapes() {
        let model = _master(this).editState.drawModel;
        if (!model)
            return null;
        return ENABLED_SHAPES_BY_TYPE[model.geometryType.toUpperCase()];
    }

    whenCancelled() {
        _master(this).whenDrawCancelled();
    }
}

interface FeatureListProps extends gws.types.ViewProps {
    features: Array<gws.types.IFeature>;
    whenFeatureTouched: (f: gws.types.IFeature) => void;
    withSearch: boolean;
    whenSearchChanged: (val: string) => void;
    searchText: string;
}

class FeatureList extends gws.View<FeatureListProps> {
    render() {
        let zoomTo = f => this.props.controller.update({
            marker: {
                features: [f],
                mode: 'zoom draw fade'
            }
        });

        let leftButton = f => {
            if (f.geometryName)
                return <components.list.Button
                    className="cmpListZoomListButton"
                    whenTouched={() => zoomTo(f)}
                />
            else
                return <components.list.Button
                    className="cmpListDefaultListButton"
                    whenTouched={() => this.props.whenFeatureTouched(f)}
                />
        }

        return <VBox>
            {this.props.withSearch && <VRow>
                <div className="modSearchBox">
                    <Row>
                        <Cell>
                            <gws.ui.Button className='searchIcon'/>
                        </Cell>
                        <Cell flex>
                            <gws.ui.TextInput
                                placeholder={this.__('editSearchPlaceholder')}
                                withClear={true}
                                value={this.props.searchText}
                                whenChanged={val => this.props.whenSearchChanged(val)}
                            />
                        </Cell>
                    </Row>
                </div>
            </VRow>}
            <VRow flex>
                <components.feature.List
                    controller={this.props.controller}
                    features={this.props.features}
                    content={f => <gws.ui.Link
                        content={f.views.title}
                        whenTouched={() => this.props.whenFeatureTouched(f)}
                    />}
                    leftButton={leftButton}
                />
            </VRow>
        </VBox>


    }
}

class Dialog extends gws.View<ViewProps> {

    render() {
        let model = this.props.editState.tableViewSelectedModel;

        if (model) {
            return <TableViewDialog {...this.props} />
        }

        let dd = this.props.editDialogData;

        if (!dd || !dd.type) {
            return null;
        }

        switch (dd.type) {
            case 'SelectModel':
                return <SelectModelDialog {...this.props} />
            case 'SelectFeature':
                return <SelectFeatureDialog {...this.props} />
            case 'DeleteFeature':
                return <DeleteFeatureDialog {...this.props} />
            case 'Error':
                return <ErrorDialog {...this.props} />
            case 'GeometryText':
                return <GeometryTextDialog {...this.props} />
        }
    }
}

class SelectModelDialog extends gws.View<ViewProps> {
    render() {
        let dd = this.props.editDialogData as SelectModelDialogData;
        let cc = _master(this);

        let cancelButton = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.closeDialog()}
        />;

        let items = dd.models.map(model => ({
            value: model.uid,
            text: model.title,
        }));

        return <gws.ui.Dialog
            className="editSelectModelDialog"
            title={this.__('editSelectModelTitle')}
            whenClosed={() => cc.closeDialog()}
            buttons={[cancelButton]}
        >
            <Form>
                <Row>
                    <Cell flex>
                        <gws.ui.List
                            items={items}
                            value={null}
                            whenChanged={v => dd.whenSelected(cc.app.modelRegistry.getModel(v))}
                        />
                    </Cell>
                </Row>
            </Form>
        </gws.ui.Dialog>;
    }
}

class SelectFeatureDialog extends gws.View<ViewProps> {
    async componentDidMount() {
        let cc = _master(this);
        let dd = this.props.editDialogData as SelectFeatureDialogData;
        await cc.featureCache.updateForModel(dd.model);
    }

    render() {
        let cc = _master(this);
        let dd = this.props.editDialogData as SelectFeatureDialogData;
        let searchText = cc.getFeatureListSearchText(dd.model.uid)
        let features = cc.featureCache.getForModel(dd.model);

        let cancelButton = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.closeDialog()}
        />;

        return <gws.ui.Dialog
            className="editSelectFeatureDialog"
            title={this.__('editSelectFeatureTitle')}
            whenClosed={() => cc.closeDialog()}
            buttons={[cancelButton]}
        >
            <FeatureList
                controller={cc}
                whenFeatureTouched={dd.whenFeatureTouched}
                whenSearchChanged={val => cc.whenFeatureListSearchChanged(dd.model, val)}
                features={features}
                searchText={searchText}
                withSearch={dd.model.supportsKeywordSearch}
            />
        </gws.ui.Dialog>;
    }
}

class DeleteFeatureDialog extends gws.View<ViewProps> {
    render() {
        let dd = this.props.editDialogData as DeleteFeatureDialogData;
        let cc = _master(this);

        return <gws.ui.Confirm
            title={this.__('editDeleteFeatureTitle')}
            text={this.__('editDeleteFeatureText').replace(/%s/, dd.feature.views.title)}
            whenConfirmed={() => dd.whenConfirmed()}
            whenRejected={() => cc.closeDialog()}
        />
    }
}

class ErrorDialog extends gws.View<ViewProps> {
    render() {
        let dd = this.props.editDialogData as ErrorDialogData;
        let cc = _master(this);

        return <gws.ui.Alert
            title={'Fehler'}
            error={dd.errorText}
            details={dd.errorDetails}
            whenClosed={() => cc.closeDialog()}
        />
    }
}

class GeometryTextDialog extends gws.View<ViewProps> {
    save() {
        let dd = this.props.editDialogData as GeometryTextDialogData;
        dd.whenSaved(dd.shape);
    }

    toWKT(shape: gws.api.base.shape.Props): string {
        let cc = _master(this);
        const wktFormat = new ol.format.WKT();
        let geom = cc.map.shape2geom(shape);
        return wktFormat.writeGeometry(geom);
    }

    updateShape(shape: gws.api.base.shape.Props) {
        let cc = _master(this);
        let dd = this.props.editDialogData as GeometryTextDialogData;

        cc.update({
            editDialogData: {...dd, shape}
        });
    }

    updateFromWKT(wkt: string) {
        let cc = _master(this);
        const wktFormat = new ol.format.WKT();
        let geom = wktFormat.readGeometry(wkt);
        let shape = cc.map.geom2shape(geom);
        this.updateShape(shape);
    }

    updatePointCoordinate(index: number, value: string) {
        let n = parseFloat(value);
        if (Number.isNaN(n)) {
            return
        }

        let dd = this.props.editDialogData as GeometryTextDialogData;
        let shape = {
            crs: dd.shape.crs,
            geometry: {
                type: dd.shape.geometry.type,
                coordinates: [...dd.shape.geometry.coordinates]
            }
        };
        shape.geometry.coordinates[index] = n;
        this.updateShape(shape);
    }


    render() {
        let dd = this.props.editDialogData as GeometryTextDialogData;
        let cc = _master(this);

        let okButton = <gws.ui.Button
            {...gws.lib.cls('editSaveButton', 'isActive')}
            tooltip={this.__('editSave')}
            whenTouched={() => this.save()}
        />


        let cancelButton = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.closeDialog()}
        />;


        return <gws.ui.Dialog
            className="editGeometryTextDialog"
            title={this.__('editGeometryTextTitle')}
            whenClosed={() => cc.closeDialog()}
            buttons={[okButton, cancelButton]}
        >
            <Form>
                {dd.shape && dd.shape.geometry.type === 'Point' && <Row>
                    <Cell>
                        <gws.ui.TextInput
                            label="X"
                            value={cc.map.formatCoordinate(dd.shape.geometry.coordinates[0])}
                            whenChanged={v => this.updatePointCoordinate(0, v)}
                            whenEntered={() => this.save()}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.TextInput
                            label="Y"
                            value={cc.map.formatCoordinate(dd.shape.geometry.coordinates[1])}
                            whenChanged={v => this.updatePointCoordinate(1, v)}
                            whenEntered={() => this.save()}
                        />
                    </Cell>
                </Row>}
                <Row>
                    <Cell flex>
                        <gws.ui.TextArea
                            label="WKT"
                            value={this.toWKT(dd.shape)}
                            whenChanged={v => this.updateFromWKT(v)}
                            height={200}
                        />
                    </Cell>
                </Row>
            </Form>
        </gws.ui.Dialog>;
    }
}

class TableViewDialog extends gws.View<ViewProps> {
    tableRef: React.RefObject<any>;

    constructor(props) {
        super(props);
        this.tableRef = React.createRef();
    }

    async saveFeature(feature) {
        let cc = _master(this);

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
        let cc = _master(this);
        let features = cc.featureCache.getForModel(model);
        let rows = []

        for (let feature of features) {
            let row = this.makeRow(feature)
            rows.push(row)
        }

        return rows;
    }


    makeRow(feature: gws.types.IFeature): TableViewRow {
        let cc = _master(this);
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

    makeSelectedRow(feature: gws.types.IFeature): TableViewRow {
        let cc = _master(this);
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
        let cc = _master(this);
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
        let cc = _master(this);
        cc.updateEditState({
            tableViewSelectedModel: null,
            tableViewSelectedFeature: null,
            tableViewRows: null,
        })


    }

    async whenCellTouched(evt) {
        let cc = _master(this);
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
        let cc = _master(this);
        feature.editAttribute(field.name, value);
        cc.updateEditState();
    }

    async whenWidgetEntered(feature: gws.types.IFeature, field: gws.types.IModelField, value: any) {
        let cc = _master(this);
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
        let cc = _master(this);

        let ok = await this.saveFeature(feature);
        if (!ok)
            return
        cc.updateEditState({
            tableViewSelectedFeature: null,
        })


    }


    whenResetButtonTouched(feature) {
        let cc = _master(this);

        feature.resetEdits();
        cc.updateEditState()
    }

    whenOpenFormButtonTouched(feature) {
        let cc = _master(this);
        cc.updateEditState({
            tableViewSelectedModel: null,
            tableViewSelectedFeature: null,
            tableViewRows: null,
            sidebarSelectedFeature: feature,
        })
    }

    whenCancelButtonTouched(feature) {
        let cc = _master(this);
        cc.updateEditState({
            tableViewSelectedFeature: null,
            serverError: '',
            formErrors: null,
        })
    }

    whenDeleteButtonTouched(feature) {
    }

    async whenNewButtonTouched() {
        let cc = _master(this);
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
        let cc = _master(this);
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
        let cc = _master(this);

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
        let cc = _master(this);
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
        let cc = _master(this);
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


export class ModelsTab extends gws.View<ViewProps> {
    async whenItemTouched(model: gws.types.IModel) {
        _master(this).selectModelInSidebar(model)
    }

    async whenRightButtonTouched(model: gws.types.IModel) {
        _master(this).selectModelInTableView(model)
    }

    render() {
        let cc = _master(this);
        let items = cc.models.filter(m => m.isEditable);

        items.sort((a, b) => a.title.localeCompare(b.title));

        if (gws.lib.isEmpty(items)) {
            return <sidebar.EmptyTab>
                {this.__('editNoLayer')}
            </sidebar.EmptyTab>;
        }

        return <sidebar.Tab className="editSidebar">
            <sidebar.TabHeader>
                <Row>
                    <Cell>
                        <gws.ui.Title content={this.__('editTitle')}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <VBox>
                    <VRow flex>
                        <components.list.List
                            controller={this.props.controller}
                            items={items}
                            content={model => <gws.ui.Link
                                whenTouched={() => this.whenItemTouched(model)}
                                content={model.title}
                            />}
                            uid={model => model.uid}
                            leftButton={model => <components.list.Button
                                className="editModelButton"
                                tooltip={this.__('editOpenModel')}
                                whenTouched={() => this.whenItemTouched(model)}
                            />}
                            rightButton={model => model.hasTableView && <components.list.Button
                                className="editTableViewButton"
                                tooltip={this.__('editTableViewButton')}
                                whenTouched={() => this.whenRightButtonTouched(model)}
                            />}
                        />
                    </VRow>
                </VBox>
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

export class ListTab extends gws.View<ViewProps> {
    async whenFeatureTouched(feature: gws.types.IFeature) {
        let cc = _master(this);
        let loaded = await cc.featureCache.loadOne(feature);
        if (loaded) {
            cc.selectFeatureInSidebar(loaded);
            cc.panToFeature(loaded);
        }
    }

    whenSearchChanged(val) {
        let cc = _master(this);
        let es = cc.editState;
        let model = es.sidebarSelectedModel;
        cc.whenFeatureListSearchChanged(model, val);
    }

    whenModelsButtonTouched() {
        let cc = _master(this);
        cc.unselectModels();
        cc.updateEditState({featureHistory: []});
    }

    whenTableViewButtonTouched() {
        let cc = _master(this);
        let es = cc.editState;
        let model = es.sidebarSelectedModel;
        _master(this).selectModelInTableView(model)
    }

    async whenNewButtonTouched() {
        let cc = _master(this);
        let feature = await cc.createFeature(cc.editState.sidebarSelectedModel);
        cc.selectFeatureInSidebar(feature);
    }

    whenNewGeometryButtonTouched() {
        let cc = _master(this);
        cc.updateEditState({
            drawModel: cc.editState.sidebarSelectedModel,
            drawFeature: null,
        });
        cc.app.startTool('Tool.Edit.Draw')
    }

    whenNewPointGeometryTextButtonTouched() {
        let cc = _master(this);
        let shape = {
            crs: cc.map.crs,
            geometry: {
                type: 'Point',
                coordinates: [0, 0]
            }
        }
        cc.showDialog({
            type: 'GeometryText',
            shape,
            whenSaved: shape => this.whenNewPointGeometrySaved(shape),
        });
    }

    async whenNewPointGeometrySaved(shape) {
        let cc = _master(this);

        let feature = await cc.createFeature(
            cc.editState.sidebarSelectedModel,
            null,
            cc.map.shape2geom(shape)
        )
        cc.selectFeatureInSidebar(feature);
        cc.closeDialog();
    }

    async componentDidMount() {
        let cc = _master(this);
        let es = cc.editState;
        await cc.featureCache.updateForModel(es.sidebarSelectedModel);
    }

    render() {
        let cc = _master(this);
        let es = this.props.editState;
        let model = es.sidebarSelectedModel;
        let features = cc.featureCache.getForModel(model);
        let searchText = es.featureListSearchText[model.uid] || '';

        let hasGeom = false;
        let hasGeomText = false;

        for (let fld of model.fields) {
            if (fld.name === model.geometryName) {
                hasGeom = true;
                hasGeomText = fld.widgetProps.type === 'geometry' && fld.widgetProps.withText;
                break;
            }
        }

        return <sidebar.Tab className="editSidebar">
            <sidebar.TabHeader>
                <Row>
                    <Cell>
                        <gws.ui.Title content={model.title}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <FeatureList
                    controller={cc}
                    whenFeatureTouched={f => this.whenFeatureTouched(f)}
                    whenSearchChanged={val => this.whenSearchChanged(val)}
                    features={features}
                    searchText={searchText}
                    withSearch={model.supportsKeywordSearch}
                />
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        {...gws.lib.cls('editModelListAuxButton')}
                        tooltip={this.__('editModelListAuxButton')}
                        whenTouched={() => this.whenModelsButtonTouched()}
                    />
                    {model.hasTableView && <sidebar.AuxButton
                        {...gws.lib.cls('editTableViewAuxButton')}
                        tooltip={this.__('editTableViewAuxButton')}
                        whenTouched={() => this.whenTableViewButtonTouched()}
                    />}
                    <Cell flex/>
                    {model.canCreate && hasGeomText && model.geometryType === gws.api.core.GeometryType.point && <sidebar.AuxButton
                        {...gws.lib.cls('editNewPointGeometryText')}
                        tooltip={this.__('editNewPointGeometryText')}
                        whenTouched={() => this.whenNewPointGeometryTextButtonTouched()}
                    />}
                    {model.canCreate && hasGeom && <sidebar.AuxButton
                        {...gws.lib.cls('editDrawAuxButton', this.props.appActiveTool === 'Tool.Edit.Draw' && 'isActive')}
                        tooltip={this.__('editDrawAuxButton')}
                        whenTouched={() => this.whenNewGeometryButtonTouched()}
                    />}
                    {model.canCreate && !hasGeom && <sidebar.AuxButton
                        {...gws.lib.cls('editNewAuxButton')}
                        tooltip={this.__('editNewAuxButton')}
                        whenTouched={() => this.whenNewButtonTouched()}
                    />}
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>

        </sidebar.Tab>

    }
}

export class FormTab extends gws.View<ViewProps> {
    async whenSaveButtonTouched(feature: gws.types.IFeature) {
        let cc = _master(this);
        let ok = await cc.saveFeatureInSidebar(feature);
        if (ok) {
            await cc.closeForm();
        }
    }

    whenDeleteButtonTouched(feature: gws.types.IFeature) {
        let cc = _master(this);
        cc.showDialog({
            type: 'DeleteFeature',
            feature,
            whenConfirmed: () => this.whenDeleteConfirmed(feature),
        })
    }

    async whenDeleteConfirmed(feature: gws.types.IFeature) {
        let cc = _master(this);
        let ok = await cc.deleteFeature(feature);
        if (ok) {
            await cc.closeDialog();
            await cc.closeForm();
        }
    }

    whenResetButtonTouched(feature: gws.types.IFeature) {
        let cc = _master(this);
        let es = cc.editState;
        feature.resetEdits();
        cc.updateEditState();
    }

    async whenCancelButtonTouched() {
        let cc = _master(this);
        await cc.closeForm();
    }

    whenWidgetChanged(feature: gws.types.IFeature, field: gws.types.IModelField, value: any) {
        let cc = _master(this);
        feature.editAttribute(field.name, value);
        cc.updateEditState();
    }

    whenWidgetEntered(feature: gws.types.IFeature, field: gws.types.IModelField, value: any) {
        let cc = _master(this);
        feature.editAttribute(field.name, value);
        cc.updateEditState();
        this.whenSaveButtonTouched(feature);
    }

    async componentDidMount() {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.sidebarSelectedFeature;

        cc.updateEditState({formErrors: []});

        for (let fld of sf.model.fields) {
            await cc.initWidget(fld);
        }
    }

    render() {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.sidebarSelectedFeature;
        let values = sf.currentAttributes();
        let widgets = [];
        let geomWidget = null;

        for (let fld of sf.model.fields) {
            let w = cc.createWidget(
                gws.types.ModelWidgetMode.form,
                fld,
                sf,
                values,
                this.whenWidgetChanged.bind(this),
                this.whenWidgetEntered.bind(this),
            );
            if (w && fld.widgetProps.type === 'geometry' && !fld.widgetProps.isInline && !geomWidget) {
                geomWidget = w;
                w = null;
            }
            widgets.push(w);
        }

        let isDirty = sf.isNew || sf.isDirty;

        return <sidebar.Tab className="editSidebar editSidebarFormTab">
            <sidebar.TabHeader>
                <Row>
                    <Cell flex>
                        <gws.ui.Title content={sf.views.title}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <VBox>
                    <VRow flex>
                        <Cell flex>
                            <Form>
                                <components.Form
                                    controller={this.props.controller}
                                    feature={sf}
                                    values={values}
                                    model={sf.model}
                                    errors={es.formErrors}
                                    widgets={widgets}
                                />
                            </Form>
                        </Cell>
                    </VRow>
                    <VRow>
                        <Row>
                            {geomWidget && <Cell>{geomWidget}</Cell>}
                            <Cell flex/>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.lib.cls('editSaveButton', isDirty && 'isActive')}
                                    tooltip={this.__('editSave')}
                                    whenTouched={() => this.whenSaveButtonTouched(sf)}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.lib.cls('editResetButton', isDirty && 'isActive')}
                                    tooltip={this.__('editReset')}
                                    whenTouched={() => this.whenResetButtonTouched(sf)}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.lib.cls('editCancelButton')}
                                    tooltip={this.__('editCancel')}
                                    whenTouched={() => this.whenCancelButtonTouched()}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    className="editDeleteButton"
                                    tooltip={this.__('editDelete')}
                                    whenTouched={() => this.whenDeleteButtonTouched(sf)}
                                />
                            </Cell>
                        </Row>
                    </VRow>
                </VBox>
            </sidebar.TabBody>

        </sidebar.Tab>
    }

}

class WidgetHelper {
    app: gws.types.IApplication

    constructor(controller: gws.types.IController) {
        this.app = controller.app;
    }

    async init(field: gws.types.IModelField) {
    }

    setProps(feature: gws.types.IFeature, field: gws.types.IModelField, props: gws.types.Dict) {
    }
}

export class GeometryWidgetHelper extends WidgetHelper {
    setProps(feature, field, props) {
        props.whenNewButtonTouched = () => this.whenNewButtonTouched(feature, field);
        props.whenEditButtonTouched = () => this.whenEditButtonTouched(feature, field);
        props.whenEditTextButtonTouched = () => this.whenEditTextButtonTouched(feature, field);
    }

    whenNewButtonTouched(feature: gws.types.IFeature, field: gws.types.IModelField) {
        let cc = _master(this);
        cc.updateEditState({
            drawModel: feature.model,
            drawFeature: feature,
        });
        cc.app.startTool('Tool.Edit.Draw');
    }

    whenEditButtonTouched(feature: gws.types.IFeature, field: gws.types.IModelField) {
        let cc = _master(this);
        cc.zoomToFeature(feature);
        cc.app.startTool('Tool.Edit.Pointer');
    }

    whenEditTextButtonTouched(feature: gws.types.IFeature, field: gws.types.IModelField) {
        let cc = _master(this);
        cc.showDialog({
            type: 'GeometryText',
            shape: feature.getAttribute(field.name),
            whenSaved: shape => this.whenEditTextSaved(feature, field, shape),
        });
    }

    async whenEditTextSaved(feature: gws.types.IFeature, field: gws.types.IModelField, shape: gws.api.base.shape.Props) {
        let cc = _master(this);
        await cc.closeDialog();
        feature.setShape(shape);
        await cc.whenModifyEnded(feature);
    }
}

class FeatureSelectWidgetHelper extends WidgetHelper {
    async init(field) {
        let cc = _master(this);
        await cc.featureCache.updateRelatableForField(field)
    }

    setProps(feature, field, props) {
        let cc = _master(this);
        props.features = cc.featureCache.getRelatableForField(field);
    }
}

class FeatureSuggestWidgetHelper extends WidgetHelper {
    async init(field) {
        let cc = _master(this);
        let searchText = cc.getFeatureListSearchText(field.uid);

        if (searchText) {
            await cc.featureCache.updateRelatableForField(field)
        }
    }

    setProps(feature, field, props) {
        let cc = _master(this);
        let searchText = cc.getFeatureListSearchText(field.uid);

        props.features = searchText ? cc.featureCache.getRelatableForField(field) : [];
        props.searchText = searchText;
        props.whenSearchChanged = val => this.whenSearchChanged(field, val);
    }

    whenSearchChanged(field, val: string) {
        let cc = _master(this);
        cc.updateFeatureListSearchText(field.uid, val);
        if (val) {
            clearTimeout(cc.searchTimer);
            cc.searchTimer = Number(setTimeout(
                () => cc.featureCache.updateRelatableForField(field),
                SEARCH_TIMEOUT
            ));
        }
    }
}

class FeatureListWidgetHelper extends WidgetHelper {
    async init(field: gws.types.IModelField) {
    }

    setProps(feature, field: gws.types.IModelField, props) {
        props.whenNewButtonTouched = () => this.whenNewButtonTouched(feature, field);
        props.whenLinkButtonTouched = () => this.whenLinkButtonTouched(feature, field);
        props.whenEditButtonTouched = r => this.whenEditButtonTouched(feature, field, r);
        props.whenUnlinkButtonTouched = f => this.whenUnlinkButtonTouched(feature, field, f);
        // props.whenDeleteButtonTouched = r => this.whenDeleteButtonTouched(field, r);
    }

    async whenNewButtonTouched(feature, field: gws.types.IModelField) {
        let cc = _master(this);
        let relatedModels = field.relatedModels();

        if (relatedModels.length === 1) {
            return this.whenModelForNewSelected(feature, field, relatedModels[0]);
        }
        cc.showDialog({
            type: 'SelectModel',
            models: relatedModels,
            whenSelected: model => this.whenModelForNewSelected(feature, field, model),
        });
    }

    async whenLinkButtonTouched(feature, field: gws.types.IModelField) {
        let cc = _master(this);
        let relatedModels = field.relatedModels();

        if (relatedModels.length === 1) {
            return this.whenModelForLinkSelected(feature, field, relatedModels[0]);
        }
        cc.showDialog({
            type: 'SelectModel',
            models: relatedModels,
            whenSelected: model => this.whenModelForLinkSelected(feature, field, model),
        });
    }


    async whenModelForNewSelected(feature, field: gws.types.IModelField, model: gws.types.IModel) {
        let cc = _master(this);

        let initFeature = model.featureWithAttributes({})
        initFeature.createWithFeatures = [feature]

        let res = await this.app.server.editInitFeature({
            modelUid: model.uid,
            feature: initFeature.getProps(1),
        });

        let newFeature = cc.app.modelRegistry.featureFromProps(res.feature);
        newFeature.isNew = true;
        newFeature.createWithFeatures = [feature]

        cc.pushFeature(feature);
        cc.selectFeatureInSidebar(newFeature);
        await cc.closeDialog();
    }


    async whenModelForLinkSelected(feature, field: gws.types.IModelField, model: gws.types.IModel) {
        let cc = _master(this);

        await cc.featureCache.updateForModel(model);

        cc.showDialog({
            type: 'SelectFeature',
            model: model,
            field,
            whenFeatureTouched: r => this.whenLinkedFeatureSelected(feature, field, r),
        });

    }

    whenLinkedFeatureSelected(feature, field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = _master(this);

        field.addRelatedFeature(feature, relatedFeature);

        cc.closeDialog();
        cc.updateEditState();
    }

    whenUnlinkButtonTouched(feature, field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = _master(this);

        field.removeRelatedFeature(feature, relatedFeature);

        cc.closeDialog();
        cc.updateEditState();
    }


    async whenEditButtonTouched(feature, field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = _master(this);

        let loaded = await cc.featureCache.loadOne(relatedFeature);
        if (loaded) {
            cc.updateEditState({isWaiting: true, sidebarSelectedFeature: null});
            gws.lib.nextTick(() => {
                cc.updateEditState({isWaiting: false});
                cc.pushFeature(feature);
                cc.selectFeatureInSidebar(loaded);
                cc.panToFeature(loaded);
            })
        }
    }

    whenDeleteButtonTouched(feature, field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = _master(this);

        cc.showDialog({
            type: 'DeleteFeature',
            feature: relatedFeature,
            whenConfirmed: () => this.whenDeleteConfirmed(feature, field, relatedFeature),
        })
    }

    async whenDeleteConfirmed(feature, field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = _master(this);

        let ok = await cc.deleteFeature(relatedFeature);

        if (ok) {
            let atts = feature.currentAttributes();
            let flist = cc.removeFeature(atts[field.name], relatedFeature);
            feature.editAttribute(field.name, flist);
        }

        await cc.closeDialog();
    }

}

export class SidebarView extends gws.View<ViewProps> {
    render() {
        let es = this.props.editState;

        if (es.isWaiting)
            return <gws.ui.Loader/>;

        if (es.sidebarSelectedFeature)
            return <FormTab {...this.props} />;

        if (es.sidebarSelectedModel)
            return <ListTab {...this.props} />;

        return <ModelsTab {...this.props} />;
    }
}

export class Sidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'editSidebarIcon';

    get tooltip() {
        return this.__('editSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarView, StoreKeys)
        );
    }
}

class EditToolbarButton extends toolbar.Button {
    iconClass = 'editToolbarButton';
    tool = 'Tool.Edit.Pointer';

    get tooltip() {
        return this.__('editToolbarButton');
    }

}

class EditLayer extends gws.map.layer.FeatureLayer {
    controller: Controller;
    cssSelector = '.editFeature'

    get printPlane() {
        return null;
    }
}

export class FeatureCache {
    app: gws.types.IApplication

    constructor(controller: gws.types.IController) {
        this.app = controller.app;
    }

    getForModel(model: gws.types.IModel): Array<gws.types.IFeature> {
        let cc = _master(this);
        let es = cc.editState;
        let key = 'model:' + model.uid
        return this.get(key)
    }

    async updateForModel(model) {
        let cc = _master(this);
        let es = cc.editState;
        let features = await this.loadMany(model, cc.getFeatureListSearchText(model.uid));
        let key = 'model:' + model.uid;
        this.checkAndStore(key, features);
    }

    getRelatableForField(field) {
        let key = 'field:' + field.uid;
        return this.get(key);
    }

    async updateRelatableForField(field: gws.types.IModelField) {
        let cc = _master(this);
        let searchText = cc.getFeatureListSearchText(field.uid);

        let request: gws.api.base.edit.action.GetRelatableFeaturesRequest = {
            modelUid: field.model.uid,
            fieldName: field.name,
            keyword: searchText || '',
            extent: cc.map.bbox,
        }
        let res = await cc.app.server.editGetRelatableFeatures(request);
        let features = cc.app.modelRegistry.featureListFromProps(res.features);
        let key = 'field:' + field.uid;
        this.checkAndStore(key, features);
    }

    async loadMany(model, searchText) {
        let cc = _master(this);
        let ls = model.loadingStrategy;

        let request: gws.api.base.edit.action.GetFeaturesRequest = {
            modelUids: [model.uid],
            keyword: searchText || '',
        }

        if (ls === gws.api.core.FeatureLoadingStrategy.lazy && !searchText) {
            return [];
        }
        if (ls === gws.api.core.FeatureLoadingStrategy.bbox) {
            request.extent = cc.map.bbox;
        }

        let res = await cc.app.server.editGetFeatures(request);
        return model.featureListFromProps(res.features);
    }

    async loadOne(feature: gws.types.IFeature) {
        let cc = _master(this);

        let res = await cc.app.server.editGetFeature({
            modelUid: feature.model.uid,
            featureUid: feature.uid,
        });

        return cc.app.modelRegistry.featureFromProps(res.feature);
    }

    checkAndStore(key: string, features: Array<gws.types.IFeature>) {
        let cc = _master(this);

        let fmap = new Map();

        for (let f of features) {
            fmap.set(f.uid, f);
        }

        for (let f of this.get(key)) {
            if (f.isDirty) {
                fmap.set(f.uid, f);
            }
        }

        this.store(key, [...fmap.values()]);
    }

    drop(uid: string) {
        let cc = _master(this);

        let es = cc.editState;
        let fc = {...es.featureCache};
        delete fc[uid];
        cc.updateEditState({
            featureCache: fc
        });
    }

    clear() {
        let cc = _master(this);

        cc.updateEditState({
            featureCache: {}
        });
    }

    store(key, features) {
        let cc = _master(this);

        let es = cc.editState;
        cc.updateEditState({
            featureCache: {
                ...(es.featureCache || {}),
                [key]: features,
            }
        });
    }

    get(key: string): Array<gws.types.IFeature> {
        let cc = _master(this);

        let es = cc.editState;
        return es.featureCache[key] || [];
    }
}

export class Controller extends gws.Controller {
    uid = MASTER;
    editLayer: EditLayer;
    models: Array<gws.types.IModel>
    setup: gws.api.base.edit.action.Props;

    widgetHelpers: { [key: string]: WidgetHelper } = {
        'geometry': new GeometryWidgetHelper(this),
        'featureList': new FeatureListWidgetHelper(this),
        'fileList': new FeatureListWidgetHelper(this),
        'featureSelect': new FeatureSelectWidgetHelper(this),
        'featureSuggest': new FeatureSuggestWidgetHelper(this),
    }

    featureCache = new FeatureCache(this);

    async init() {
        await super.init();
        this.models = [];

        this.updateEditState({
            featureListSearchText: {},
            featureHistory: [],
            featureCache: {},
        });

        this.setup = this.app.actionProps('edit');
        if (!this.setup) {
            this.updateObject('sidebarHiddenItems', {'Sidebar.Edit': true});
            this.updateObject('toolbarHiddenItems', {'Toolbar.Edit': true});
            return;
        }


        let res = await this.app.server.editGetModels({});
        if (!res.error) {
            for (let props of res.models) {
                this.models.push(this.app.modelRegistry.addModel(props));
            }
        }

        if (gws.lib.isEmpty(this.models)) {
            this.updateObject('sidebarHiddenItems', {'Sidebar.Edit': true});
            this.updateObject('toolbarHiddenItems', {'Toolbar.Edit': true});
            return;
        }

        this.editLayer = this.map.addServiceLayer(new EditLayer(this.map, {
            uid: '_edit',
        }));
        this.editLayer.controller = this;

        this.app.whenCalled('editModel', args => {
            this.selectModelInSidebar(args.model);
            this.update({
                sidebarActiveTab: 'Sidebar.Edit',
            });
        });

        this.app.whenChanged('mapViewState', () =>
            this.whenMapStateChanged());
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(Dialog, StoreKeys));
    }

    //

    async whenMapStateChanged() {
        let es = this.editState;
        let model = es.sidebarSelectedModel;

        if (!model)
            return;

        if (model.loadingStrategy == gws.api.core.FeatureLoadingStrategy.bbox) {
            await this.featureCache.updateForModel(model);
        }
    }

    searchTimer = 0

    async whenFeatureListSearchChanged(model: gws.types.IModel, val: string) {
        this.updateFeatureListSearchText(model.uid, val);
        clearTimeout(this.searchTimer);
        this.searchTimer = Number(setTimeout(
            () => this.featureCache.updateForModel(model),
            SEARCH_TIMEOUT
        ));
    }


    async whenPointerDownAtCoordinate(coord: ol.Coordinate) {
        let pt = new ol.geom.Point(coord);
        let models;

        // @TODO should be an option whether to query the selected model only or all of them

        if (this.editState.sidebarSelectedModel)
            models = [this.editState.sidebarSelectedModel]
        else
            models = this.models.filter(m => !m.layer || !m.layer.hidden);

        let res = await this.app.server.editGetFeatures({
            shapes: [this.map.geom2shape(pt)],
            modelUids: models.map(m => m.uid),
            resolution: this.map.viewState.resolution,
        });

        if (gws.lib.isEmpty(res.features)) {
            this.unselectFeatures();
            console.log('whenPointerDownAtCoordinate: no feature')
            return;
        }

        let loaded = this.app.modelRegistry.featureFromProps(res.features[0]);
        let sf = this.editState.sidebarSelectedFeature;

        if (sf && sf.model === loaded.model && sf.uid === loaded.uid) {
            console.log('whenPointerDownAtCoordinate: same feature')
            return;
        }

        this.selectFeatureInSidebar(loaded);
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Edit'});

        console.log('whenPointerDownAtCoordinate: found feature', loaded)

    }

    _geometrySaveTimer = 0;


    async whenModifyEnded(feature: gws.types.IFeature) {
        let save = async () => {
            if (feature.isNew) {
                return;
            }

            console.log('whenModifyEnded: begin save')
            this.app.stopTool('Tool.Edit.Pointer');

            let ok = await this.saveFeatureInSidebar(feature);
            if (ok) {
                await this.featureCache.updateForModel(feature.model);
            }

            this.app.startTool('Tool.Edit.Pointer');
            console.log('whenModifyEnded: end save')
        }

        let schedule = () => {
            clearTimeout(this._geometrySaveTimer);
            this._geometrySaveTimer = Number(setTimeout(save, GEOMETRY_TIMEOUT));
        }

        schedule();
    }

    async whenDrawEnded(oFeature: ol.Feature) {
        let df = this.editState.drawFeature;

        if (df) {
            df.setGeometry(oFeature.getGeometry());
            this.selectFeatureInSidebar(df);
        } else {
            let feature = await this.createFeature(
                this.editState.drawModel,
                null,
                oFeature.getGeometry()
            )
            this.selectFeatureInSidebar(feature);
        }
        this.updateEditState({
            drawModel: null,
            drawFeature: null,
        });
    }

    whenDrawCancelled() {
        this.updateEditState({
            drawModel: null,
            drawFeature: null,
        });
        this.app.stopTool('Tool.Edit.Draw');
    }

    async initWidget(field: gws.types.IModelField) {
        let cc = _master(this);

        let controller = cc.widgetControllerForField(field);
        if (!controller)
            return;

        let p = field.widgetProps;
        if (cc.widgetHelpers[p.type]) {
            await cc.widgetHelpers[p.type].init(field);
        }
    }


    createWidget(
        mode: gws.types.ModelWidgetMode,
        field: gws.types.IModelField,
        feature: gws.types.IFeature,
        values: gws.types.Dict,
        whenChanged,
        whenEntered,
    ): React.ReactElement | null {
        let cc = _master(this);
        let widgetProps = field.widgetProps;

        if (widgetProps.type === 'hidden') {
            return null;
        }

        let controller = cc.widgetControllerForField(field);
        if (!controller) {
            return null;
        }

        let props: gws.types.Dict = {
            controller,
            feature,
            field,
            widgetProps,
            values,
            whenChanged: val => whenChanged(feature, field, val),
            whenEntered: val => whenEntered(feature, field, val),
        }

        if (cc.widgetHelpers[field.widgetProps.type]) {
            cc.widgetHelpers[field.widgetProps.type].setProps(feature, field, props);
        }

        return controller[mode + 'View'](props)
    }


    //

    showDialog(dd: DialogData) {
        this.update({editDialogData: dd})
    }

    async closeDialog() {
        this.updateEditState({isWaiting: true});
        await gws.lib.sleep(2);
        this.update({editDialogData: null});
        this.updateEditState({isWaiting: false});
    }

    async closeForm() {
        this.updateEditState({isWaiting: true});
        await gws.lib.sleep(2);

        let ok = await this.popFeature();

        if (!ok) {
            let sf = this.editState.sidebarSelectedFeature;
            this.unselectFeatures();
            if (sf) {
                this.selectModelInSidebar(sf.model);
            }
        }

        this.updateEditState({isWaiting: false});
    }

    //

    selectModelInSidebar(model: gws.types.IModel) {
        this.updateEditState({sidebarSelectedModel: model});
        this.featureCache.drop(model.uid);
    }

    selectModelInTableView(model: gws.types.IModel) {
        this.updateEditState({tableViewSelectedModel: model});
        this.featureCache.drop(model.uid);
    }

    unselectModels() {
        this.updateEditState({
            sidebarSelectedModel: null,
            tableViewSelectedModel: null,
        });
        this.featureCache.clear();
    }

    selectFeatureInSidebar(feature: gws.types.IFeature) {
        this.updateEditState({sidebarSelectedFeature: feature});
        if (feature.geometry) {
            this.app.startTool('Tool.Edit.Pointer');
        }
    }

    unselectFeatures() {
        this.updateEditState({
            sidebarSelectedFeature: null,
            tableViewSelectedFeature: null,
            featureHistory: [],
        })
    }

    pushFeature(feature: gws.types.IFeature) {
        let hist = this.removeFeature(this.editState.featureHistory, feature);
        this.updateEditState({
            featureHistory: [...hist, feature],
        });
    }

    async popFeature() {
        let hist = this.editState.featureHistory || [];
        if (hist.length === 0) {
            return false;
        }

        let last = hist[hist.length - 1];
        let loaded = await this.featureCache.loadOne(last);
        if (loaded) {
            this.updateEditState({
                featureHistory: hist.slice(0, -1)
            });
            this.selectFeatureInSidebar(loaded);
            return true;
        }

        this.updateEditState({
            featureHistory: [],
        });
        return false;
    }

    panToFeature(feature) {
        if (!feature.geometry)
            return;
        this.update({
            marker: {
                features: [feature],
                mode: 'pan',
            }
        })
    }

    zoomToFeature(feature) {
        if (!feature.geometry)
            return;
        this.update({
            marker: {
                features: [feature],
                mode: 'zoom',
            }
        })
    }

    widgetControllerForField(field: gws.types.IModelField): gws.types.IModelWidget | null {
        let p = field.widgetProps;
        if (!p)
            return null;

        let tag = 'ModelWidget.' + p.type;
        return (
            this.app.controllerByTag(tag) ||
            this.app.createControllerFromConfig(this, {tag})
        ) as gws.types.IModelWidget;
    }


    //


    //

    getModel(modelUid) {
        return this.app.modelRegistry.getModel(modelUid)
    }

    removeFeature(flist: Array<gws.types.IFeature>, feature: gws.types.IFeature) {
        return (flist || []).filter(f => f.model !== feature.model || f.uid !== feature.uid);
    }

    //

    async saveFeatureInSidebar(feature: gws.types.IFeature) {
        let ok = await this.saveFeature(feature, feature.model.fields)

        if (!ok && this.editState.serverError) {
            this.showDialog({
                type: 'Error',
                errorText: this.editState.serverError,
                errorDetails: '',
            })
            return false;
        }

        if (!ok) {
            return false;
        }

        this.featureCache.clear();
        this.map.forceUpdate();

        return true
    }

    async saveFeature(feature: gws.types.IFeature, fields: Array<gws.types.IModelField>) {
        this.updateEditState({
            serverError: '',
            formErrors: null,
        });

        let model = feature.model;

        let atts = feature.currentAttributes();
        let attsToWrite = {}

        for (let field of fields) {
            let v = atts[field.name]
            if (!gws.lib.isEmpty(v))
                attsToWrite[field.name] = await this.serializeValue(field, v);
        }

        if (!attsToWrite[feature.model.uidName])
            attsToWrite[feature.model.uidName] = feature.uid

        let featureToWrite = feature.clone();
        featureToWrite.attributes = attsToWrite;

        let res = await this.app.server.editWriteFeature({
            modelUid: model.uid,
            feature: featureToWrite.getProps(1),
        }, {binaryRequest: true});

        if (res.error) {
            this.updateEditState({serverError: this.__('editSaveErrorText')});
            return false
        }

        let formErrors = {};
        for (let e of res.validationErrors) {
            let msg = e['message'];
            if (msg.startsWith(DEFAULT_MESSAGE_PREFIX))
                msg = this.__(msg)
            formErrors[e.fieldName] = msg;
        }

        if (!gws.lib.isEmpty(formErrors)) {
            this.updateEditState({formErrors});
            return false
        }

        let savedFeature = this.app.modelRegistry.featureFromProps(res.feature);
        feature.copyFrom(savedFeature)

        return true;
    }

    async createFeature(model: gws.types.IModel, attributes?: object, geometry?: ol.geom.Geometry): Promise<gws.types.IFeature> {
        attributes = attributes || {};

        if (geometry) {
            attributes[model.geometryName] = this.map.geom2shape(geometry)
        }

        let featureToInit = model.featureWithAttributes(attributes);
        featureToInit.isNew = true;

        let res = await this.app.server.editInitFeature({
            modelUid: model.uid,
            feature: featureToInit.getProps(1),
        });

        let newFeature = model.featureFromProps(res.feature);
        newFeature.isNew = true;

        return newFeature;
    }

    async deleteFeature(feature: gws.types.IFeature) {

        let res = await this.app.server.editDeleteFeature({
            modelUid: feature.model.uid,
            feature: feature.getProps()
        });

        if (res.error) {
            this.showDialog({
                type: 'Error',
                errorText: this.__('editDeleteErrorText'),
                errorDetails: '',
            })
            return false;
        }

        this.featureCache.clear();
        this.map.forceUpdate();

        return true;
    }

    async serializeValue(field: gws.types.IModelField, val) {
        if (field.attributeType === gws.api.core.AttributeType.file) {
            if (val instanceof FileList) {
                let content: Uint8Array = await gws.lib.readFile(val[0]);
                return {name: val[0].name, content} as gws.types.ClientFileProps
            }
            return null
        }
        return val;
    }

    //

    updateFeatureListSearchText(key: string, val: string) {
        let es = this.editState;

        this.updateEditState({
            featureListSearchText: {
                ...es.featureListSearchText,
                [key]: val,
            }
        });
    }

    getFeatureListSearchText(key: string) {
        let es = this.editState;
        return es.featureListSearchText[key] || '';
    }


    get editState(): EditState {
        return this.getValue('editState') || {};
    }

    updateEditState(es: Partial<EditState> = null) {
        let editState = {
            ...this.editState,
            ...(es || {})
        }

        this.update({editState})

        if (this.editLayer) {

            this.editLayer.clear();
            let f = editState.sidebarSelectedFeature;

            if (f) {
                this.editLayer.addFeature(f);
                this.app.map.focusFeature(f);
                f.redraw()
            }
        }


    }


}

gws.registerTags({
    'Shared.Edit': Controller,
    'Sidebar.Edit': Sidebar,
    'Tool.Edit.Pointer': PointerTool,
    'Tool.Edit.Draw': DrawTool,
    'Toolbar.Edit': EditToolbarButton,
});
