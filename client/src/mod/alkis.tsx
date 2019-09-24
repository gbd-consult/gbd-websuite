import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

import * as lens from './common/lens';
import * as sidebar from './common/sidebar';
import * as storage from './common/storage';

const STORAGE_CATEGORY = 'alkis.features';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Alkis';

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as AlkisController;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as AlkisController;
}

const EXPORT_GROUPS = [
    ['base', 'Basisdaten'],
    ['lage', 'Lage'],
    ['gebaeude', 'Gebäude'],
    ['buchung', 'Buchungsblatt'],
    ['eigentuemer', 'Eigentümer'],
    ['nutzung', 'Nutzung'],
];

const EXPORT_PATH = 'fs_info.csv';

type AlkisAlkisTabName = 'form' | 'list' | 'details' | 'error' | 'selection' | 'export';

interface AlkisViewProps extends gws.types.ViewProps {
    controller: AlkisController;

    alkisTab: AlkisAlkisTabName;

    alkisFsSetup: gws.api.AlkisFsSetupResponse;

    alkisFsLoading: boolean;
    alkisFsError: string;

    alkisFsExportGroups: Array<string>;
    alkisFsExportFeatures: Array<gws.types.IMapFeature>;

    alkisFsParams: gws.api.AlkisFsQueryParams;

    alkisFsStrassen: Array<gws.ui.MenuItem>;
    alkisFsGemarkungen: Array<gws.ui.MenuItem>;

    alkisFsResults: Array<gws.types.IMapFeature>;
    alkisFsResultCount: number;

    alkisFsDetailsFeature: gws.types.IMapFeature;
    alkisFsDetailsText: string;

    alkisFsSelection: Array<gws.types.IMapFeature>;

    appActiveTool: string;

    features?: Array<gws.types.IMapFeature>;
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
    'alkisFsParams',
    'alkisFsStrassen',
    'alkisFsGemarkungen',
    'alkisFsResults',
    'alkisFsResultCount',
    'alkisFsDetailsFeature',
    'alkisFsDetailsText',
    'alkisFsSelection',
    'appActiveTool',
];

function featureIn(fs: Array<gws.types.IMapFeature>, f: gws.types.IMapFeature) {
    return fs.some(g => g.uid === f.uid);
}

class AlkisExportAuxButton extends gws.View<AlkisViewProps> {
    render() {
        if (!this.props.alkisFsSetup.ui.useExport || this.props.features.length === 0)
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
        if (!this.props.alkisFsSetup.printTemplate || this.props.features.length === 0)
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
        if (!this.props.alkisFsSetup.printTemplate || this.props.features.length === 0)
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
        if (!this.props.alkisFsSetup.ui.useSelect)
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
        if (!this.props.alkisFsSetup.ui.useSelect)
            return null;

        let master = _master(this),
            feature = this.props.features[0];

        if (master.isSelected(feature))
            return <sidebar.AuxButton
                className="modAlkisUnselectAuxButton"
                whenTouched={() => master.unselect([feature])}
                tooltip={_master(this).STRINGS.unselect}
            />
        else
            return <sidebar.AuxButton
                className="modAlkisSelectAuxButton"
                whenTouched={() => master.select([feature])}
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
        if (!this.props.alkisFsSetup.ui.useSelect)
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
        let master = _master(this),
            setup = this.props.alkisFsSetup;

        let boundTo = param => ({
            value: this.props.alkisFsParams[param],
            whenChanged: value => master.updateFsParams({[param]: value}),
            whenEntered: () => master.search()
        });

        let nameShowMode = '';

        if (setup.withEigentuemer)
            if (!setup.withControl)
                nameShowMode = 'enabled';
            else if (this.props.alkisFsParams.wantEigentuemer)
                nameShowMode = 'enabled';
            else
                nameShowMode = '';

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

            <Row>
                <Cell flex>
                    <gws.ui.Select
                        placeholder={_master(this).STRINGS.gemarkung}
                        items={this.props.alkisFsGemarkungen}
                        value={this.props.alkisFsParams.gemarkungOrGemeindeUid}
                        whenChanged={value => master.whenGemarkungChanged(value)}
                        withSearch
                        withClear
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    <gws.ui.Select
                        placeholder={_master(this).STRINGS.strasse}
                        items={this.props.alkisFsStrassen}
                        {...boundTo('strasse')}
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
                    />
                </Cell>
            </Row>}
            {setup.withControl && this.props.alkisFsParams.wantEigentuemer && <Row>
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
                    <gws.ui.IconButton
                        {...gws.tools.cls('modAlkisSearchSubmitButton')}
                        tooltip={_master(this).STRINGS.submitButton}
                        whenTouched={() => master.formSearch()}
                    />
                </Cell>
                {setup.ui.searchSelection && <Cell>
                    <gws.ui.IconButton
                        {...gws.tools.cls('modAlkisSearchSelectionButton')}
                        tooltip={_master(this).STRINGS.selectionSearchButton}
                        whenTouched={() => master.selectionSearch()}
                    />
                </Cell>}
                {setup.ui.searchSpatial && <Cell>
                    <gws.ui.IconButton
                        {...gws.tools.cls('modAlkisSearchLensButton', this.props.appActiveTool === 'Tool.Alkis.Lens' && 'isActive')}
                        tooltip={_master(this).STRINGS.lensButton}
                        whenTouched={() => master.startLens()}
                    />
                </Cell>}
                {setup.ui.usePick && <Cell>
                    <gws.ui.IconButton
                        {...gws.tools.cls('modAlkisPickButton', this.props.appActiveTool === 'Tool.Alkis.Pick' && 'isActive')}
                        tooltip={_master(this).STRINGS.pickButton}
                        whenTouched={() => master.startPick()}
                    />
                </Cell>}
                <Cell>
                    <gws.ui.IconButton
                        {...gws.tools.cls('modAlkisSearchResetButton')}
                        tooltip={_master(this).STRINGS.resetButton}
                        whenTouched={() => master.reset()}
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
        let master = _master(this);

        let rightButton = f => master.isSelected(f)
            ? <gws.components.list.Button
                className="modAlkisUnselectListButton"
                whenTouched={() => master.unselect([f])}
            />

            : <gws.components.list.Button
                className="modAlkisSelectListButton"
                whenTouched={() => master.select([f])}
            />
        ;

        if (!this.props.alkisFsSetup.ui.useSelect)
            rightButton = null;

        let content = f => <gws.ui.Link
            whenTouched={() => master.showDetails(f)}
            content={f.props.teaser}
        />;

        return <gws.components.feature.List
            controller={master}
            features={this.props.features}
            content={content}
            isSelected={f => this.props.showSelection && master.isSelected(f)}
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
        let master = _master(this);
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
                    <storage.ReadAuxButton
                        controller={this.props.controller}
                        category={STORAGE_CATEGORY}
                        whenDone={data => master.loadSelection(data.features)}
                    />
                    {hasFeatures && <storage.WriteAuxButton
                        controller={this.props.controller}
                        category={STORAGE_CATEGORY}
                        data={{features: this.props.alkisFsSelection.map(f => f.props)}}
                    />}
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
                <gws.components.Description content={feature.props.description}/>
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
        let groups = this.props.alkisFsExportGroups;

        let changed = (group, value) => _master(this).update({
            alkisFsExportGroups: groups.filter(g => g !== group).concat(value ? [group] : [])
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
                                {EXPORT_GROUPS.map(([group, name]) => {
                                    if (group === 'buchung' && !this.props.alkisFsSetup.withBuchung)
                                        return null;
                                    if (group === 'eigentuemer' && !this.props.alkisFsSetup.withEigentuemer)
                                        return null;
                                    return <gws.ui.Toggle
                                        key={group}
                                        type="checkbox"
                                        label={name}
                                        value={groups.indexOf(group) >= 0}
                                        whenChanged={value => changed(group, value)}
                                    />
                                })
                                }
                            </Cell>
                        </Row>
                        <Row>
                            <Cell flex/>
                            <Cell width={120}>
                                <gws.ui.TextButton
                                    primary
                                    whenTouched={() => _master(this).submitExport(this.props.alkisFsExportFeatures)}
                                >{_master(this).STRINGS.exportButton}</gws.ui.TextButton>
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
        if (!_master(this).setup)
            return <sidebar.EmptyTab>
                {_master(this).STRINGS.noData}
            </sidebar.EmptyTab>;

        return this.createElement(
            this.connect(AlkisSidebarView, AlkisStoreKeys));
    }

}

class AlkisLensTool extends lens.LensTool {
    title = this.__('modAlkisLensTitle');

    async runSearch(geom) {
        await _master(this).search({shapes: [this.map.geom2shape(geom)]});
    }

    stop() {
        super.stop();
        _master(this).updateFsParams({shapes: null});
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
    setup: gws.api.AlkisFsSetupResponse;
    history: Array<string>;
    selectionLayer: gws.types.IMapFeatureLayer;

    STRINGS = null;

    updateFsParams(obj) {
        this.update({
            alkisFsParams: {
                ...this.getValue('alkisFsParams'),
                ...obj
            }
        });
    }

    async init() {

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
        };

        let res = await this.app.server.alkisFsSetup({});

        if (res.error) {
            this.setup = null;
            return;
        }

        this.setup = res;

        if (this.setup.ui.gemarkungListMode === 'tree') {
            this.STRINGS.gemarkung = this.__('modAlkisGemeindeGemarkung')
        }

        this.history = [];

        this.update({
            alkisFsSetup: this.setup,

            alkisTab: 'form',
            alkisFsLoading: false,

            alkisFsParams: {},

            alkisFsExportGroups: ['base'],

            alkisFsStrassen: [],
            alkisFsGemarkungen: this.makeGemarkungList(),

            alkisFsSearchResponse: null,
            alkisFsDetailsResponse: null,
            alkisFsSelection: [],

        });

        this.app.whenLoaded(() => this.urlSearch());
    }

    makeGemarkungList(): Array<gws.ui.MenuItem> {
        let sort = a => a.sort((x, y) => x.text.localeCompare(y.text));
        let gs = this.setup.gemarkungen;

        switch (this.setup.ui.gemarkungListMode) {

            case 'plain':
                return sort(gs.map(g => ({
                    text: g.gemarkung,
                    value: 'gemarkung:' + g.gemarkungUid,
                })));

            case 'combined':
                return sort(gs.map(g => ({
                    text: g.gemarkung + ' (' + g.gemeinde
                        .replace(/^Stadt\s+/, '')
                        .replace(/\(.+/, '')
                        .trim()
                    + ')',
                    value: 'gemarkung:' + g.gemarkungUid,
                })));

            case 'tree':
                let gemeinde = {};

                gs.forEach(g => gemeinde[g.gemeindeUid] = {
                    text: g.gemeinde,
                    uid: g.gemeindeUid,
                    item: {
                        text: g.gemeinde,
                        value: 'gemeinde:' + g.gemeindeUid,
                        level: 1,
                    }
                });

                let a = [];

                sort(Object.values(gemeinde)).forEach(gem => {
                    a.push(gem.item, ...sort(gs
                        .filter(g => g.gemeindeUid === gem.uid)
                        .map(g => ({
                                text: g.gemarkung,
                                value: 'gemarkung:' + g.gemarkungUid,
                                level: 2,
                            })
                        )));
                });

                return a;
        }

    }

    async whenGemarkungChanged(value) {
        let strassen = [];

        if (value) {
            let res = await this.app.server.alkisFsStrassen({
                gemarkungOrGemeindeUid: value,
            });

            if (!res.error)
                strassen = res.strassen.map(s => ({text: s, value: s}));
        }

        this.updateFsParams({
            gemarkungOrGemeindeUid: value,
            strasse: ''
        });

        this.update({
            alkisFsStrassen: strassen
        });
    };

    selectionSearch() {
        let geoms = this.selectionGeometries();
        if (geoms) {
            this.updateFsParams({shapes: geoms.map(g => this.map.geom2shape(g))});
            this.search();
        }
    }

    selectionGeometries() {
        let sel = this.getValue('selectFeatures') as Array<gws.types.IMapFeature>;

        if (sel)
            return sel.map(f => f.geometry);

        let m = this.getValue('marker');
        if (m && m.features) {
            let gs = gws.tools.compact(m.features.map((f: gws.types.IMapFeature) => f.geometry));
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

        let res = await this.app.server.alkisFsSearch({
            shapes: [this.map.geom2shape(pt)],
        });

        if (res.error) {
            return;
        }

        let features = this.map.readFeatures(res.features);

        this.select(features);
        this.goTo('selection');
    }

    async search(params?: object) {

        if (params)
            this.updateFsParams(params);

        this.update({alkisFsLoading: true});

        let res = await this.app.server.alkisFsSearch({
            ...this.getValue('alkisFsParams'),
        });

        if (res.error) {
            let msg = this.STRINGS.errorGeneric;

            if (res.error.status === 400) {
                msg = this.STRINGS.errorControl
            }

            if (res.error.status === 409) {
                msg = this.STRINGS.errorTooMany.replace(/\$1/g, this.getValue('alkisFsSetup').limit);
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
            await this.showDetails(features[0], false);
        } else {
            this.goTo('list');
        }

        this.update({alkisFsLoading: false});
    }

    async urlSearch() {
        let p, params = null;

        p = this.app.urlParams['alkisFs'];
        if (p) {
            params = {alkisFs: p};
        }

        p = this.app.urlParams['alkisAd'];
        if (p) {
            params = {alkisAd: p};
        }

        if (!params)
            return;

        let res = await this.app.server.alkisFsSearch(params);

        if (res.error) {
            return false;
        }

        let features = this.map.readFeatures(res.features);

        if (features.length > 0)
            this.update({
                marker: {
                    features: [features[0]],
                    mode: 'draw zoom',
                },
                infoboxContent: <gws.components.Infobox
                    controller={this}>{features[0].props.teaser}</gws.components.Infobox>,
            });

    }

    paramsForFeatures(fs: Array<gws.types.IMapFeature>) {
        let queryParams: gws.api.AlkisFsQueryParams = this.getValue('alkisFsParams');
        return {
            wantEigentuemer: queryParams.wantEigentuemer,
            controlInput: queryParams.controlInput,
            fsUids: fs.map(f => f.uid),
        }

    }

    async showDetails(f: gws.types.IMapFeature, highlight = true) {
        let q = this.paramsForFeatures([f]);
        let res = await this.app.server.alkisFsDetails(q);
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

    async startPrint(fs: Array<gws.types.IMapFeature>) {
        this.update({
            printJob: {state: gws.api.JobState.init},
            marker: null,
        });

        let quality = 0;
        let level = this.setup.printTemplate.qualityLevels[quality];
        let dpi = level ? level.dpi : 0;

        let base = await this.map.printParams(null, dpi);
        let printParams: gws.api.PrintParams = {
            ...base,
            templateUid: this.setup.printTemplate.uid,
            quality
        };

        let q = {
            ...this.paramsForFeatures(fs),
            printParams,
            highlightStyle: this.map.getStyleFromSelector('.modMarkerFeature').props,
        };

        this.update({
            printJob: await this.app.server.alkisFsPrint(q),
            printSnapshotMode: false,
        });
    }

    highlightMany(fs: Array<gws.types.IMapFeature>) {
        this.update({
            marker: {
                features: fs,
                mode: 'zoom draw'
            }
        })
    }

    highlight(f: gws.types.IMapFeature) {
        this.update({
            marker: {
                features: [f],
                mode: 'zoom draw'
            }
        })
    }

    isSelected(f: gws.types.IMapFeature) {
        let sel = this.getValue('alkisFsSelection') || [];
        return sel.length && featureIn(sel, f);

    }

    select(fs: Array<gws.types.IMapFeature>) {
        if (!this.selectionLayer) {
            this.selectionLayer = this.map.addServiceLayer(new gws.map.layer.FeatureLayer(this.map, {
                uid: '_select',
                style: this.map.getStyleFromSelector('.modAlkisSelectFeature'),
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

    unselect(fs: Array<gws.types.IMapFeature>) {
        let sel = this.getValue('alkisFsSelection') || [];

        this.update({
            alkisFsSelection: sel.filter(f => !featureIn(fs, f))
        });

        this.selectionLayer.clear();
        this.selectionLayer.addFeatures(this.getValue('alkisFsSelection'));
    }

    loadSelection(fs) {
        this.clearSelection();
        this.select(this.map.readFeatures(fs));
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
            alkisFsParams: {},
            alkisFsStrassen: [],
        });
        this.clearResults();
        this.stopTools();
        this.goTo('form')
    }

    async startExport(fs: Array<gws.types.IMapFeature>) {
        this.update({
            alkisFsExportFeatures: fs
        });
        this.goTo('export')
    }

    async submitExport(fs: Array<gws.types.IMapFeature>) {
        let groups = this.getValue('alkisFsExportGroups');

        let q = {
            ...this.paramsForFeatures(fs),
            groups: EXPORT_GROUPS.map(grp => grp[0]).filter(g => groups.indexOf(g) >= 0),
        };
        let res = await this.app.server.alkisFsExport(q);

        let a = document.createElement('a');
        a.href = res.url;
        a.download = EXPORT_PATH;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    goTo(tab) {
        if (this.history[this.history.length - 1] !== tab)
            this.history.push(tab);
        this.update({
            alkisTab: tab
        });
        if (tab === 'form') {
            this.updateFsParams({controlInput: ''});
        }
    }

}

export const tags = {
    [MASTER]: AlkisController,
    'Sidebar.Alkis': AlkisSidebar,
    'Tool.Alkis.Lens': AlkisLensTool,
    'Tool.Alkis.Pick': AlkisPickTool,
};
