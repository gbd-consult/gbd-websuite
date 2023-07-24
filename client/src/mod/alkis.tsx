import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

import * as lens from './lens';
import * as sidebar from './sidebar';
import * as storage from './storage';

const STORAGE_CATEGORY = 'Alkis';
const MASTER = 'Shared.Alkis';

let {Form, Row, Cell} = gws.ui.Layout;

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as AlkisController;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as AlkisController;
}

const EXPORT_PATH = 'fs_info.csv';

type AlkisAlkisTabName = 'form' | 'list' | 'details' | 'error' | 'selection' | 'export';

const _PREFIX_GEMARKUNG = '@gemarkung';
const _PREFIX_GEMEINDE = '@gemeinde';

interface FormValues {
    bblatt?: string;
    controlInput?: string;
    flaecheBis?: string;
    flaecheVon?: string;
    gemarkungUid?: string;
    gemeindeUid?: string;
    hausnummer?: string;
    name?: string;
    strasseUid?: string;
    vnum?: string;
    vorname?: string;
    wantEigentuemer?: boolean;
    shapes?: Array<gws.api.ShapeProps>;
}

interface Strasse {
    name: string;
    uid: string;
    gemarkungUid: string;
    gemeindeUid: string;
}

interface Toponyms {
    gemarkungen: Array<gws.api.AlkissearchToponymGemarkung>;
    gemeinden: Array<gws.api.AlkissearchToponymGemeinde>;
    gemarkungIndex: gws.types.Dict,
    gemeindeIndex: gws.types.Dict,
    strassen: Array<Strasse>;
}

interface AlkisViewProps extends gws.types.ViewProps {
    controller: AlkisController;

    alkisTab: AlkisAlkisTabName;

    alkisFsLoading: boolean;
    alkisFsError: string;

    alkisFsExportGroups: Array<string>;
    alkisFsExportFeatures: Array<gws.types.IFeature>;

    alkisFsFormValues: FormValues;

    alkisFsGemarkungListItems: Array<gws.ui.ListItem>;
    alkisFsStrasseListItems: Array<gws.ui.ListItem>;

    alkisFsResults: Array<gws.types.IFeature>;
    alkisFsResultCount: number;

    alkisFsDetailsFeature: gws.types.IFeature;
    alkisFsDetailsText: string;

    alkisFsSelection: Array<gws.types.IFeature>;

    appActiveTool: string;

    features?: Array<gws.types.IFeature>;
    showSelection: boolean;

}

interface AlkisMessageViewProps extends AlkisViewProps {
    message?: string;
    error?: string;
    withFormLink?: boolean;
};

const AlkisStoreKeys = [
    'alkisTab',
    'alkisFsSetup',
    'alkisFsLoading',
    'alkisFsError',
    'alkisFsExportGroups',
    'alkisFsExportFeatures',
    'alkisFsFormValues',
    'alkisFsStrasseListItems',
    'alkisFsGemarkungListItems',
    'alkisFsResults',
    'alkisFsResultCount',
    'alkisFsDetailsFeature',
    'alkisFsDetailsText',
    'alkisFsSelection',
    'appActiveTool',
];

function featureIn(fs: Array<gws.types.IFeature>, f: gws.types.IFeature) {
    return fs.some(g => g.uid === f.uid);
}

class AlkisExportAuxButton extends gws.View<AlkisViewProps> {
    render() {
        let mm = _master(this);
        if (!mm.setup.ui.useExport || this.props.features.length === 0)
            return null;
        return <sidebar.AuxButton
            className="modAlkisExportAuxButton"
            whenTouched={() => _master(this).startExport(this.props.features)}
            tooltip={_master(this).STRINGS.exportTitle}
        />;
    }
}

class AlkisPrintAuxButton extends gws.View<AlkisViewProps> {
    render() {
        let mm = _master(this);
        if (!mm.setup.printTemplate || this.props.features.length === 0)
            return null;
        return <sidebar.AuxButton
            className="modAlkisPrintAuxButton"
            whenTouched={() => _master(this).startPrint(this.props.features)}
            tooltip={_master(this).STRINGS.print}
        />;
    }
}

class AlkisHighlightAuxButton extends gws.View<AlkisViewProps> {
    render() {
        let mm = _master(this);
        if (!mm.setup.printTemplate || this.props.features.length === 0)
            return null;
        return <sidebar.AuxButton
            className="modAlkisHighlightAuxButton"
            whenTouched={() => _master(this).highlightMany(this.props.features)}
            tooltip={_master(this).STRINGS.highlight}
        />
    }
}

class AlkisSelectAuxButton extends gws.View<AlkisViewProps> {
    render() {
        let mm = _master(this);
        if (!mm.setup.ui.useSelect)
            return null;
        return <sidebar.AuxButton
            className="modAlkisSelectAuxButton"
            whenTouched={() => _master(this).select(this.props.features)}
            tooltip={_master(this).STRINGS.selectAll}
        />
    }
}

class AlkisToggleAuxButton extends gws.View<AlkisViewProps> {
    render() {
        let mm = _master(this);

        if (!mm.setup.ui.useSelect)
            return null;

        let feature = this.props.features[0];

        if (mm.isSelected(feature))
            return <sidebar.AuxButton
                className="modAlkisUnselectAuxButton"
                whenTouched={() => mm.unselect([feature])}
                tooltip={_master(this).STRINGS.unselect}
            />
        else
            return <sidebar.AuxButton
                className="modAlkisSelectAuxButton"
                whenTouched={() => mm.select([feature])}
                tooltip={_master(this).STRINGS.select}
            />
    }
}

class AlkisFormAuxButton extends gws.View<AlkisViewProps> {
    render() {
        return <sidebar.AuxButton
            {...gws.tools.cls('modAlkisFormAuxButton', this.props.alkisTab === 'form' && 'isActive')}
            whenTouched={() => _master(this).goTo('form')}
            tooltip={_master(this).STRINGS.gotoForm}
        />
    }
}

class AlkisListAuxButton extends gws.View<AlkisViewProps> {
    render() {
        return <sidebar.AuxButton
            {...gws.tools.cls('modAlkisListAuxButton', this.props.alkisTab === 'list' && 'isActive')}
            whenTouched={() => _master(this).goTo('list')}
            tooltip={_master(this).STRINGS.gotoList}
        />
    }
}

class AlkisSelectionAuxButton extends gws.View<AlkisViewProps> {
    render() {
        let mm = _master(this);

        if (!mm.setup.ui.useSelect)
            return null;

        let sel = this.props.alkisFsSelection || [];

        return <sidebar.AuxButton
            {...gws.tools.cls('modAlkisSelectionAuxButton', this.props.alkisTab === 'selection' && 'isActive')}
            badge={sel.length ? String(sel.length) : null}
            whenTouched={() => _master(this).goTo('selection')}
            tooltip={_master(this).STRINGS.gotoSelection}
        />
    }
}

class AlkisClearAuxButton extends gws.View<AlkisViewProps> {
    render() {
        if (this.props.alkisFsSelection.length === 0)
            return null;

        return <sidebar.AuxButton
            className="modAlkisClearAuxButton"
            whenTouched={() => _master(this).clearSelection()}
            tooltip={_master(this).STRINGS.clearSelection}
        />
    }
}

class AlkisResetAuxButton extends gws.View<AlkisViewProps> {
    render() {
        return <sidebar.AuxButton
            className="modAlkisResetAuxButton"
            whenTouched={() => _master(this).reset()}
            tooltip={_master(this).STRINGS.resetButton}
        />
    }
}

class AlkisNavigation extends gws.View<AlkisViewProps> {
    render() {
        return <React.Fragment>
            <AlkisFormAuxButton {...this.props}/>
            <AlkisListAuxButton {...this.props}/>
            <AlkisSelectionAuxButton {...this.props}/>
        </React.Fragment>
    }
}

class AlkisLoaderTab extends gws.View<AlkisViewProps> {
    render() {
        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={_master(this).STRINGS.formTitle}/>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                <div className="modAlkisLoading">
                    {_master(this).STRINGS.loading}
                    <gws.ui.Loader/>
                </div>
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

class AlkisMessageTab extends gws.View<AlkisMessageViewProps> {
    render() {
        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={_master(this).STRINGS.formTitle}/>
            </sidebar.TabHeader>

            <sidebar.EmptyTabBody>
                {this.props.error && <gws.ui.Error text={this.props.error}/>}
                {this.props.message && <gws.ui.Text content={this.props.message}/>}
                {this.props.withFormLink &&
                <a onClick={() => _master(this).goTo('form')}>{_master(this).STRINGS.backToForm}</a>}
            </sidebar.EmptyTabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <AlkisNavigation {...this.props}/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>

        </sidebar.Tab>
    }
}

class AlkisSearchForm extends gws.View<AlkisViewProps> {

    render() {
        let mm = _master(this),
            setup = mm.setup,
            form = this.props.alkisFsFormValues;

        let boundTo = key => ({
            value: form[key],
            whenChanged: value => mm.updateForm({[key]: value}),
            whenEntered: () => mm.search()
        });

        let nameShowMode = '';

        if (setup.withEigentuemer) {
            if (!setup.withControl)
                nameShowMode = 'enabled';
            else if (form.wantEigentuemer)
                nameShowMode = 'enabled';
            else
                nameShowMode = '';
        }

        let gemarkungListValue = '';

        if (form.gemarkungUid)
            gemarkungListValue = _PREFIX_GEMARKUNG + ':' + form.gemarkungUid;
        else if (form.gemeindeUid)
            gemarkungListValue = _PREFIX_GEMEINDE + ':' + form.gemeindeUid;

        let strasseSearchMode = {
            anySubstring: setup.ui.strasseSearchMode === gws.api.AlkissearchUiStrasseSearchMode.any,
            withExtra: true,
            caseSensitive: false,
        };

        return <Form>
            {nameShowMode && <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={_master(this).STRINGS.vorname}
                        disabled={nameShowMode === 'disabled'}
                        {...boundTo('vorname')}
                        withClear
                    />
                </Cell>
            </Row>}

            {nameShowMode && <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={_master(this).STRINGS.nachname}
                        disabled={nameShowMode === 'disabled'}
                        {...boundTo('name')}
                        withClear
                    />
                </Cell>
            </Row>}

            {setup.ui.gemarkungListMode !== gws.api.AlkissearchUiGemarkungListMode.none && <Row>
                <Cell flex>
                    <gws.ui.Select
                        placeholder={_master(this).STRINGS.gemarkung}
                        items={this.props.alkisFsGemarkungListItems}
                        value={gemarkungListValue}
                        whenChanged={value => mm.whenGemarkungChanged(value)}
                        withSearch
                        withClear
                    />
                </Cell>
            </Row>}

            <Row>
                <Cell flex>
                    <gws.ui.Select
                        placeholder={_master(this).STRINGS.strasse}
                        items={this.props.alkisFsStrasseListItems}
                        {...boundTo('strasseUid')}
                        searchMode={strasseSearchMode}
                        withSearch
                        withClear
                    />
                </Cell>
                <Cell width={90}>
                    <gws.ui.TextInput
                        placeholder={_master(this).STRINGS.nr}
                        {...boundTo('hausnummer')}
                        withClear
                    />
                </Cell>
            </Row>

            <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={setup.withFlurnummer ? _master(this).STRINGS.vnumFlur : _master(this).STRINGS.vnum}
                        {...boundTo('vnum')}
                        withClear
                    />
                </Cell>
            </Row>

            <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={_master(this).STRINGS.areaFrom}
                        {...boundTo('flaecheVon')}
                        withClear
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={_master(this).STRINGS.areaTo}
                        {...boundTo('flaecheBis')}
                        withClear
                    />
                </Cell>
            </Row>

            {setup.withBuchung && <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={_master(this).STRINGS.bblatt}
                        {...boundTo('bblatt')}
                        withClear
                    />
                </Cell>
            </Row>}

            {setup.withControl && <Row className='modAlkisControlToggle'>
                <Cell flex>
                    <gws.ui.Toggle
                        type="checkbox"
                        {...boundTo('wantEigentuemer')}
                        label={_master(this).STRINGS.wantEigentuemer}
                        inline={true}
                    />
                </Cell>
            </Row>}

            {setup.withControl && form.wantEigentuemer && <Row>
                <Cell flex>
                    <gws.ui.TextArea
                        {...boundTo('controlInput')}
                        placeholder={_master(this).STRINGS.controlInput}
                    />
                </Cell>
            </Row>}

            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.Button
                        {...gws.tools.cls('modAlkisSearchSubmitButton')}
                        tooltip={_master(this).STRINGS.submitButton}
                        whenTouched={() => mm.formSearch()}
                    />
                </Cell>
                {setup.ui.searchSelection && <Cell>
                    <gws.ui.Button
                        {...gws.tools.cls('modAlkisSearchSelectionButton')}
                        tooltip={_master(this).STRINGS.selectionSearchButton}
                        whenTouched={() => mm.selectionSearch()}
                    />
                </Cell>}
                {setup.ui.searchSpatial && <Cell>
                    <gws.ui.Button
                        {...gws.tools.cls('modAlkisSearchLensButton', this.props.appActiveTool === 'Tool.Alkis.Lens' && 'isActive')}
                        tooltip={_master(this).STRINGS.lensButton}
                        whenTouched={() => mm.startLens()}
                    />
                </Cell>}
                {setup.ui.usePick && <Cell>
                    <gws.ui.Button
                        {...gws.tools.cls('modAlkisPickButton', this.props.appActiveTool === 'Tool.Alkis.Pick' && 'isActive')}
                        tooltip={_master(this).STRINGS.pickButton}
                        whenTouched={() => mm.startPick()}
                    />
                </Cell>}
                <Cell>
                    <gws.ui.Button
                        {...gws.tools.cls('modAlkisSearchResetButton')}
                        tooltip={_master(this).STRINGS.resetButton}
                        whenTouched={() => mm.reset()}
                    />
                </Cell>
            </Row>
        </Form>;
    }
}

class AlkisFormTab extends gws.View<AlkisViewProps> {
    render() {
        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={_master(this).STRINGS.formTitle}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <AlkisSearchForm {...this.props} />
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <AlkisNavigation {...this.props}/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>


        </sidebar.Tab>
    }
}

class AlkisFeatureList extends gws.View<AlkisViewProps> {

    render() {
        let mm = _master(this);

        let rightButton = f => mm.isSelected(f)
            ? <gws.components.list.Button
                className="modAlkisUnselectListButton"
                whenTouched={() => mm.unselect([f])}
            />

            : <gws.components.list.Button
                className="modAlkisSelectListButton"
                whenTouched={() => mm.select([f])}
            />
        ;

        if (!mm.setup.ui.useSelect)
            rightButton = null;

        let content = f => <gws.ui.Link
            whenTouched={() => mm.showDetails(f)}
            content={f.elements.teaser}
        />;

        return <gws.components.feature.List
            controller={mm}
            features={this.props.features}
            content={content}
            isSelected={f => this.props.showSelection && mm.isSelected(f)}
            rightButton={rightButton}
            withZoom
        />

    }

}

class AlkisListTab extends gws.View<AlkisViewProps> {
    title() {
        let total = this.props.alkisFsResultCount,
            disp = this.props.alkisFsResults.length;

        if (!disp)
            return _master(this).STRINGS.formTitle;

        let s = (total > disp) ? _master(this).STRINGS.searchResults2 : _master(this).STRINGS.searchResults;

        s = s.replace(/\$1/g, String(disp));
        s = s.replace(/\$2/g, String(total));

        return s;
    }

    render() {
        let features = this.props.alkisFsResults;

        if (gws.tools.empty(features)) {
            return <AlkisMessageTab
                {...this.props}
                message={_master(this).STRINGS.notFound}
                withFormLink={true}
            />;
        }

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.title()}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <AlkisFeatureList {...this.props} features={features} showSelection={true}/>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <AlkisNavigation {...this.props}/>
                    <Cell flex/>
                    <AlkisSelectAuxButton {...this.props} features={features}/>
                    <AlkisExportAuxButton {...this.props} features={features}/>
                    <AlkisPrintAuxButton {...this.props} features={features}/>
                    <AlkisResetAuxButton {...this.props} features={features}/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class AlkisSelectionTab extends gws.View<AlkisViewProps> {
    render() {
        let mm = _master(this);
        let features = this.props.alkisFsSelection;
        let hasFeatures = !gws.tools.empty(features);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={_master(this).STRINGS.selectionTitle}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {hasFeatures
                    ? <AlkisFeatureList {...this.props} features={features} showSelection={false}/>
                    : <sidebar.EmptyTabBody>{_master(this).STRINGS.noSelection}</sidebar.EmptyTabBody>
                }
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <AlkisNavigation {...this.props}/>
                    <Cell flex/>
                    {hasFeatures && <AlkisExportAuxButton {...this.props} features={features}/>}
                    {hasFeatures && <AlkisPrintAuxButton {...this.props} features={features}/>}
                    {storage.auxButtons(mm, {
                        category: STORAGE_CATEGORY,
                        hasData: hasFeatures,
                        getData: name => mm.selectionData(),
                        dataReader: (name, data) => mm.loadSelection(data)
                    })}
                    {hasFeatures && <AlkisClearAuxButton {...this.props} />}
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class AlkisDetailsTab extends gws.View<AlkisViewProps> {
    render() {
        let feature = this.props.alkisFsDetailsFeature;

        return <sidebar.Tab>

            <sidebar.TabHeader>
                <gws.ui.Title content={_master(this).STRINGS.infoTitle}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.components.Description content={feature.elements.description}/>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <AlkisNavigation {...this.props}/>
                    <Cell flex/>
                    <AlkisToggleAuxButton {...this.props} features={[feature]}/>
                    <AlkisPrintAuxButton {...this.props} features={[feature]}/>
                    <Cell>
                        <gws.components.feature.TaskButton
                            controller={this.props.controller}
                            feature={feature}
                            source="alkis"
                        />
                    </Cell>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>

        </sidebar.Tab>

    }
}

class AlkisExportTab extends gws.View<AlkisViewProps> {
    render() {
        let mm = _master(this);

        let availGroups = mm.setup.exportGroups,
            selectedGroupIds = this.props.alkisFsExportGroups;

        let changed = (gid, value) => mm.update({
            alkisFsExportGroups: selectedGroupIds.filter(g => g !== gid).concat(value ? [gid] : [])
        });

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={_master(this).STRINGS.exportTitle}/>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                <div className="modAlkisAlkisFsDetailsTabContent">
                    <Form>
                        <Row>
                            <Cell flex>
                                <gws.ui.Group vertical>
                                    {Object.keys(availGroups).map(gid => <gws.ui.Toggle
                                            key={gid}
                                            type="checkbox"
                                            inline
                                            label={availGroups[gid]}
                                            value={selectedGroupIds.includes(gid)}
                                            whenChanged={value => changed(gid, value)}
                                        />
                                    )}
                                </gws.ui.Group>
                            </Cell>
                        </Row>
                        <Row>
                            <Cell flex/>
                            <Cell width={120}>
                                <gws.ui.Button
                                    primary
                                    whenTouched={() => mm.submitExport(this.props.alkisFsExportFeatures)}
                                    label={mm.STRINGS.exportButton}
                                />
                            </Cell>
                        </Row>
                    </Form>
                </div>
            </sidebar.TabBody>
            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <AlkisNavigation {...this.props}/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>

    }
}

class AlkisSidebarView extends gws.View<AlkisViewProps> {
    render() {
        let mm = _master(this);

        if (!mm.setup)
            return <sidebar.EmptyTab>
                {mm.STRINGS.noData}
            </sidebar.EmptyTab>;

        if (this.props.alkisFsLoading)
            return <AlkisLoaderTab {...this.props}/>;

        let tab = this.props.alkisTab;

        if (!tab || tab === 'form')
            return <AlkisFormTab {...this.props} />;

        if (tab === 'list')
            return <AlkisListTab {...this.props} />;

        if (tab === 'details')
            return <AlkisDetailsTab {...this.props} />;

        if (tab === 'error')
            return <AlkisMessageTab {...this.props} error={this.props.alkisFsError} withFormLink={true}/>;

        if (tab === 'selection')
            return <AlkisSelectionTab {...this.props} />;

        if (tab === 'export')
            return <AlkisExportTab {...this.props} />;
    }
}

class AlkisSidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'modAlkisSidebarIcon';

    get tooltip() {
        return this.__('modAlkisSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(AlkisSidebarView, AlkisStoreKeys));
    }

}

class AlkisLensTool extends lens.LensTool {
    title = this.__('modAlkisLensTitle');

    async runSearch(geom) {
        _master(this).updateForm({shapes: [this.map.geom2shape(geom)]});
        await _master(this).search();
    }

    stop() {
        super.stop();
        _master(this).updateForm({shapes: null});
    }
}

class AlkisPickTool extends gws.Tool {
    start() {
        this.map.prependInteractions([
            this.map.pointerInteraction({
                whenTouched: evt => _master(this).pickTouched(evt.coordinate),
            }),
        ]);
    }

    stop() {

    }
}

class AlkisController extends gws.Controller {
    uid = MASTER;
    history: Array<string>;
    selectionLayer: gws.types.IFeatureLayer;
    setup: gws.api.AlkissearchProps;
    toponyms: Toponyms;

    STRINGS = null;

    async init() {
        this.setup = this.app.actionSetup('alkissearch');

        this.STRINGS = {

            formTitle: this.__('modAlkisFormTitle'),
            infoTitle: this.__('modAlkisInfoTitle'),
            selectionTitle: this.__('modAlkisSelectionTitle'),
            exportTitle: this.__('modAlkisExportTitle'),
            gotoForm: this.__('modAlkisGotoForm'),
            gotoList: this.__('modAlkisGotoList'),
            gotoSelection: this.__('modAlkisGotoSelection'),
            print: this.__('modAlkisPrint'),
            highlight: this.__('modAlkisHighlight'),
            selectAll: this.__('modAlkisSelectAll'),
            unselect: this.__('modAlkisUnselect'),
            select: this.__('modAlkisSelect'),
            clearSelection: this.__('modAlkisClearSelection'),
            goBack: this.__('modAlkisGoBack'),
            loadSelection: this.__('modAlkisLoadSelection'),
            saveSelection: this.__('modAlkisSaveSelection'),
            loading: this.__('modAlkisLoading'),
            backToForm: this.__('modAlkisBackToForm'),
            noData: this.__('modAlkisNoData'),
            notFound: this.__('modAlkisNotFound'),
            noSelection: this.__('modAlkisNoSelection'),
            errorGeneric: this.__('modAlkisErrorGeneric'),
            errorControl: this.__('modAlkisErrorControl'),
            errorTooMany: this.__('modAlkisErrorTooMany'),
            vorname: this.__('modAlkisVorname'),
            nachname: this.__('modAlkisNachname'),
            gemarkung: this.__('modAlkisGemarkung'),
            strasse: this.__('modAlkisStrasse'),
            nr: this.__('modAlkisNr'),
            vnum: this.__('modAlkisVnum'),
            vnumFlur: this.__('modAlkisVnumFlur'),
            areaFrom: this.__('modAlkisAreaFrom'),
            areaTo: this.__('modAlkisAreaTo'),
            bblatt: this.__('modAlkisBblatt'),
            wantEigentuemer: this.__('modAlkisWantEigentuemer'),
            controlInput: this.__('modAlkisControlInput'),
            submitButton: this.__('modAlkisSubmitButton'),
            lensButton: this.__('modAlkisLensButton'),
            pickButton: this.__('modAlkisPickButton'),
            selectionSearchButton: this.__('modAlkisSelectionSearchButton'),
            resetButton: this.__('modAlkisResetButton'),
            exportButton: this.__('modAlkisExportButton'),
            searchResults2: this.__('modAlkisSearchResults2'),
            searchResults: this.__('modAlkisSearchResults'),
            errorExport: this.__('modAlkisErrorExport'),
        };

        if (!this.setup)
            return;

        if (this.setup.ui.gemarkungListMode === 'tree') {
            this.STRINGS.gemarkung = this.__('modAlkisGemeindeGemarkung')
        }

        this.history = [];

        console.time('ALKIS:toponyms:load');

        this.toponyms = await this.loadToponyms();

        console.timeEnd('ALKIS:toponyms:load');


        this.update({
            alkisTab: 'form',
            alkisFsLoading: false,

            alkisFsFormValues: {},
            alkisFsExportGroups: [],

            alkisFsGemarkungListItems: this.gemarkungListItems(),
            alkisFsStrasseListItems: this.strasseListItems(this.toponyms.strassen),

            alkisFsSearchResponse: null,
            alkisFsDetailsResponse: null,
            alkisFsSelection: [],
        });
    }

    async loadToponyms(): Promise<Toponyms> {
        let res = await this.app.server.alkissearchGetToponyms({});

        let t: Toponyms = {
            gemeinden: res.gemeinden,
            gemarkungen: res.gemarkungen,
            gemeindeIndex: {},
            gemarkungIndex: {},
            strassen: []
        };

        for (let g of res.gemeinden) {
            t.gemeindeIndex[g.uid] = g;
        }
        for (let g of res.gemarkungen) {
            t.gemarkungIndex[g.uid] = g;
        }

        t.strassen = res.strasseNames.map((name, n) => ({
            name,
            uid: String(n),
            gemarkungUid: res.strasseGemarkungUids[n],
            gemeindeUid: t.gemarkungIndex[res.strasseGemarkungUids[n]].gemeindeUid,
        }));

        return t;
    }

    gemarkungListItems(): Array<gws.ui.ListItem> {
        let items = [];

        switch (this.setup.ui.gemarkungListMode) {

            case gws.api.AlkissearchUiGemarkungListMode.plain:
                for (let g of this.toponyms.gemarkungen) {
                    items.push({
                        text: g.name,
                        value: _PREFIX_GEMARKUNG + ':' + g.uid,
                    })
                }
                break;

            case gws.api.AlkissearchUiGemarkungListMode.combined:
                for (let g of this.toponyms.gemarkungen) {
                    items.push({
                        text: g.name,
                        extraText: this.toponyms.gemeindeIndex[g.gemeindeUid].name,
                        value: _PREFIX_GEMARKUNG + ':' + g.uid,
                    });
                }
                break;

            case gws.api.AlkissearchUiGemarkungListMode.tree:
                for (let gd of this.toponyms.gemeinden) {
                    items.push({
                        text: gd.name,
                        value: _PREFIX_GEMEINDE + ':' + gd.uid,
                        level: 1,
                    });
                    for (let gk of this.toponyms.gemarkungen) {
                        if (gk.gemeindeUid === gd.uid) {
                            items.push({
                                text: gk.name,
                                value: _PREFIX_GEMARKUNG + ':' + gk.uid,
                                level: 2,
                            });
                        }
                    }
                }
                break;
        }

        return items;
    }

    strasseListItems(strassen: Array<Strasse>): Array<gws.ui.ListItem> {
        let strasseStats = {}, ls = [];

        for (let s of strassen) {
            strasseStats[s.name] = (strasseStats[s.name] || 0) + 1;
        }

        switch (this.setup.ui.strasseListMode) {

            case gws.api.AlkissearchUiStrasseListMode.plain:
                ls = strassen.map(s => ({
                    text: s.name,
                    value: s.uid,
                    extraText: '',
                }));
                break;

            case gws.api.AlkissearchUiStrasseListMode.withGemarkung:
                ls = strassen.map(s => ({
                    text: s.name,
                    value: s.uid,
                    extraText: this.toponyms.gemarkungIndex[s.gemarkungUid].name,
                }));
                break;

            case gws.api.AlkissearchUiStrasseListMode.withGemarkungIfRepeated:
                ls = strassen.map(s => ({
                    text: s.name,
                    value: s.uid,
                    extraText: strasseStats[s.name] > 1
                        ? this.toponyms.gemarkungIndex[s.gemarkungUid].name
                        : ''
                }));
                break;

            case gws.api.AlkissearchUiStrasseListMode.withGemeinde:
                ls = strassen.map(s => ({
                    text: s.name,
                    value: s.uid,
                    extraText: this.toponyms.gemeindeIndex[s.gemeindeUid].name,
                }));
                break;

            case gws.api.AlkissearchUiStrasseListMode.withGemeindeIfRepeated:
                ls = strassen.map(s => ({
                    text: s.name,
                    value: s.uid,
                    extraText: strasseStats[s.name] > 1
                        ? this.toponyms.gemeindeIndex[s.gemeindeUid].name
                        : ''
                }));
                break;
        }

        return ls.sort((a, b) => a.text.localeCompare(b.text) || a.extraText.localeCompare(b.extraText));
    }

    async whenGemarkungChanged(value) {
        let strassen = this.toponyms.strassen,
            form: FormValues = {
                gemarkungUid: null,
                gemeindeUid: null,
                strasseUid: null,
            };

        if (value) {
            let p = value.split(':');

            if (p[0] === _PREFIX_GEMEINDE) {
                form.gemeindeUid = p[1];
                strassen = this.toponyms.strassen.filter(s => s.gemeindeUid === form.gemeindeUid);
            }
            if (p[0] === _PREFIX_GEMARKUNG) {
                form.gemarkungUid = p[1];
                form.gemeindeUid = this.toponyms.gemarkungIndex[p[1]].gemeindeUid;
                strassen = this.toponyms.strassen.filter(s => s.gemarkungUid === form.gemarkungUid);
            }
        }

        this.updateForm(form);

        this.update({
            alkisFsStrasseListItems: this.strasseListItems(strassen),
        });
    };

    selectionSearch() {
        let geoms = this.selectionGeometries();
        if (geoms) {
            this.updateForm({shapes: geoms.map(g => this.map.geom2shape(g))});
            this.search();
        }
    }

    selectionGeometries() {
        let sel = this.getValue('selectFeatures') as Array<gws.types.IFeature>;

        if (sel)
            return sel.map(f => f.geometry);

        let m = this.getValue('marker');
        if (m && m.features) {
            let gs = gws.tools.compact(m.features.map((f: gws.types.IFeature) => f.geometry));
            if (gs.length > 0) {
                return gs
            }
        }
    }

    formSearch() {
        this.stopTools();
        this.search();
        if (this.setup.ui.autoSpatialSearch)
            this.startLens();
    }

    startLens() {
        this.app.startTool('Tool.Alkis.Lens');
    }

    stopTools() {
        this.app.stopTool('Tool.Alkis.*');
    }

    startPick() {
        this.app.startTool('Tool.Alkis.Pick');
    }

    async pickTouched(coord: ol.Coordinate) {
        let pt = new ol.geom.Point(coord);

        let res = await this.app.server.alkissearchFindFlurstueck({
            shapes: [this.map.geom2shape(pt)],
        });

        if (res.error) {
            return;
        }

        let features = this.map.readFeatures(res.features);

        this.select(features);
        this.goTo('selection');
    }

    async search() {

        this.update({alkisFsLoading: true});

        let params = {...this.getValue('alkisFsFormValues')};

        if (!gws.tools.empty(params.strasseUid)) {
            let strasse = this.toponyms.strassen[Number(params.strasseUid)];
            if (strasse) {
                params.gemarkungUid = strasse.gemarkungUid;
                params.strasse = strasse.name;
            }
        }
        delete params.strasseUid;

        let res = await this.app.server.alkissearchFindFlurstueck(params);

        if (res.error) {
            let msg = this.STRINGS.errorGeneric;

            if (res.error.status === 400) {
                msg = this.STRINGS.errorControl
            }

            if (res.error.status === 409) {
                msg = this.STRINGS.errorTooMany.replace(/\$1/g, this.setup.limit);
            }

            this.update({
                alkisFsError: msg,
            });

            this.update({alkisFsLoading: false});
            return this.goTo('error');
        }

        let features = this.map.readFeatures(res.features);

        this.update({
            alkisFsResults: features,
            alkisFsResultCount: res.total,
            marker: {
                features,
                mode: 'zoom draw',
            },
            infoboxContent: null
        });

        if (features.length === 1) {
            await this.showDetails(features[0], true);
        } else {
            this.goTo('list');
        }

        this.update({alkisFsLoading: false});
    }

    paramsForFeatures(fs: Array<gws.types.IFeature>) {
        let form = this.getValue('alkisFsFormValues');
        return {
            wantEigentuemer: form.wantEigentuemer,
            controlInput: form.controlInput,
            fsUids: fs.map(f => f.uid),
        }
    }

    async showDetails(f: gws.types.IFeature, highlight = true) {
        let q = this.paramsForFeatures([f]);
        let res = await this.app.server.alkissearchGetDetails(q);
        let feature = this.map.readFeature(res.feature);

        if (f) {
            if (highlight)
                this.highlight(f);

            this.update({
                alkisFsDetailsFeature: feature,
            });

            this.goTo('details');
        }
    }

    async startPrint(fs: Array<gws.types.IFeature>) {
        this.update({
            printJob: {state: gws.api.JobState.init},
            marker: null,
        });

        let quality = 0;
        let level = this.setup.printTemplate.qualityLevels[quality];
        let dpi = level ? level.dpi : 0;

        let basicParams = await this.map.basicPrintParams(null, dpi);
        let printParams: gws.api.PrintParamsWithTemplate = {
            type: 'template',
            templateUid: this.setup.printTemplate.uid,
            quality,
            ...basicParams,
        };

        let q = {
            findParams: this.paramsForFeatures(fs),
            printParams,
            highlightStyle: this.app.style.get('.modMarkerFeature').props,
        };

        this.update({
            printJob: await this.app.server.alkissearchPrint(q),
            printSnapshotMode: false,
        });
    }

    highlightMany(fs: Array<gws.types.IFeature>) {
        this.update({
            marker: {
                features: fs,
                mode: 'zoom draw'
            }
        })
    }

    highlight(f: gws.types.IFeature) {
        this.update({
            marker: {
                features: [f],
                mode: 'zoom draw'
            }
        })
    }

    isSelected(f: gws.types.IFeature) {
        let sel = this.getValue('alkisFsSelection') || [];
        return sel.length && featureIn(sel, f);

    }

    select(fs: Array<gws.types.IFeature>) {
        if (!this.selectionLayer) {
            this.selectionLayer = this.map.addServiceLayer(new gws.map.layer.FeatureLayer(this.map, {
                uid: '_select',
                cssSelector: '.modAlkisSelectFeature',
            }));
        }

        let sel = this.getValue('alkisFsSelection') || [],
            add = [];

        fs.forEach(f => {
            if (!featureIn(sel, f))
                add.push(f)
        });

        this.update({
            alkisFsSelection: sel.concat(add)
        });

        this.selectionLayer.clear();
        this.selectionLayer.addFeatures(this.getValue('alkisFsSelection'));
    }

    unselect(fs: Array<gws.types.IFeature>) {
        let sel = this.getValue('alkisFsSelection') || [];

        this.update({
            alkisFsSelection: sel.filter(f => !featureIn(fs, f))
        });

        this.selectionLayer.clear();
        this.selectionLayer.addFeatures(this.getValue('alkisFsSelection'));
    }

    selectionData() {
        let fs = this.getValue('alkisFsSelection');
        return {
            selection: fs.map(f => f.uid)
        }
    }

    async loadSelection(data) {
        this.clearSelection();

        let res = await this.app.server.alkissearchFindFlurstueck({
            wantEigentuemer: false,
            fsUids: data.selection,
        });

        if (res.error) {
            return;
        }

        let features = this.map.readFeatures(res.features);

        this.select(features);

        this.update({
            marker: {
                features,
                mode: 'zoom fade',
            }
        });

    }

    clearSelection() {
        this.update({
            alkisFsSelection: []
        });
        this.map.removeLayer(this.selectionLayer);
        this.selectionLayer = null;
    }

    clearResults() {
        this.update({
            alkisFsResultCount: 0,
            alkisFsResults: [],
            marker: null,
        });
    }

    reset() {
        this.update({
            alkisFsFormValues: {},
            alkisFsStrasseListItems: this.strasseListItems(this.toponyms.strassen),
        });
        this.clearResults();
        this.stopTools();
        this.goTo('form')
    }

    async startExport(fs: Array<gws.types.IFeature>) {
        this.update({
            alkisFsExportFeatures: fs
        });
        this.goTo('export')
    }

    async submitExport(fs: Array<gws.types.IFeature>) {
        let q = {
            findParams: this.paramsForFeatures(fs),
            groups: this.getValue('alkisFsExportGroups'),
        };

        // NB: must use binary because csv doesn't neccessary come in utf8

        let res = await this.app.server.alkissearchExport(q, {binary: true});

        if (res.error) {
            return;
        }

        let a = document.createElement('a');
        a.href = window.URL.createObjectURL(new Blob([res.content], {type: res.mime}));
        a.download = EXPORT_PATH;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(a.href);
        document.body.removeChild(a);
    }

    goTo(tab) {
        if (this.history[this.history.length - 1] !== tab)
            this.history.push(tab);
        this.update({
            alkisTab: tab
        });
        if (tab === 'form') {
            this.updateForm({controlInput: null});
        }
    }

    updateForm(obj) {
        this.update({
            alkisFsFormValues: {
                ...this.getValue('alkisFsFormValues'),
                ...obj
            }
        });
    }
}

export const tags = {
    [MASTER]: AlkisController,
    'Sidebar.Alkis': AlkisSidebar,
    'Tool.Alkis.Lens': AlkisLensTool,
    'Tool.Alkis.Pick': AlkisPickTool,
};
