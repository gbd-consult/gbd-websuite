import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from 'gws/elements/sidebar';
import * as components from 'gws/components';
import * as lens from 'gws/elements/lens';
import * as storage from 'gws/elements/storage';

const MASTER = 'Shared.Alkis';

let {Form, Row, Cell} = gws.ui.Layout;

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as Controller;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as Controller;
}

const EXPORT_PATH = 'fs_info.csv';

type TabName = 'form' | 'list' | 'details' | 'error' | 'selection' | 'export';

const _PREFIX_GEMARKUNG = '@gemarkung';
const _PREFIX_GEMEINDE = '@gemeinde';

interface FormValues {
    bblatt?: string;
    eigentuemerControlInput?: string;
    flaecheBis?: number;
    flaecheVon?: number;
    gemarkungCode?: string;
    gemeindeCode?: string;
    hausnummer?: string;
    personName?: string;
    personVorname?: string;
    shapes?: Array<gws.api.base.shape.Props>;
    strasseCode?: string;
    fsnummer?: string;
    wantEigentuemer?: boolean;
    wantHistoryDisplay?: boolean;
    wantHistorySearch?: boolean;
}

interface Gemeinde {
    code: string;
    name: string;
}

interface Gemarkung {
    code: string;
    name: string;
    gemeindeCode: string;
}

interface Strasse {
    code: string;
    name: string;
    gemeindeCode: string;
    gemarkungCode: string;
}

interface Toponyms {
    gemeinden: Array<Gemeinde>;
    gemarkungen: Array<Gemarkung>;
    strassen: Array<Strasse>;
    gemarkungIndex: gws.types.Dict,
    gemeindeIndex: gws.types.Dict,
}

interface ViewProps extends gws.types.ViewProps {
    controller: Controller;

    alkisTab: TabName;

    alkisFsLoading: boolean;
    alkisFsError: string;

    alkisFsExportGroupIndexes: Array<number>;
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

interface MessageViewProps extends ViewProps {
    message?: string;
    error?: string;
    withFormLink?: boolean;
};

const StoreKeys = [
    'alkisTab',
    'alkisFsSetup',
    'alkisFsLoading',
    'alkisFsError',
    'alkisFsExportGroupIndexes',
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

class ExportAuxButton extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);

        if (cc.setup.exportGroups.length === 0 || this.props.features.length === 0)
            return null;
        return <sidebar.AuxButton
            className="alkisExportAuxButton"
            whenTouched={() => cc.startExport(this.props.features)}
            tooltip={cc.__('alkisExportTitle')}
        />;
    }
}

class PrintAuxButton extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);

        if (!cc.setup.printer || this.props.features.length === 0)
            return null;
        return <sidebar.AuxButton
            className="alkisPrintAuxButton"
            whenTouched={() => cc.startPrint(this.props.features)}
            tooltip={cc.__('alkisPrint')}
        />;
    }
}

class HighlightAuxButton extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);

        if (!cc.setup.printer || this.props.features.length === 0)
            return null;
        return <sidebar.AuxButton
            className="alkisHighlightAuxButton"
            whenTouched={() => cc.highlightMany(this.props.features)}
            tooltip={cc.__('alkisHighlight')}
        />
    }
}

class SelectAuxButton extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);

        if (!cc.setup.ui.useSelect)
            return null;
        return <sidebar.AuxButton
            className="alkisSelectAuxButton"
            whenTouched={() => cc.select(this.props.features)}
            tooltip={cc.__('alkisSelectAll')}
        />
    }
}

class ToggleAuxButton extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);

        if (!cc.setup.ui.useSelect)
            return null;

        let feature = this.props.features[0];

        if (cc.isSelected(feature))
            return <sidebar.AuxButton
                className="alkisUnselectAuxButton"
                whenTouched={() => cc.unselect([feature])}
                tooltip={cc.__('alkisUnselect')}
            />
        else
            return <sidebar.AuxButton
                className="alkisSelectAuxButton"
                whenTouched={() => cc.select([feature])}
                tooltip={cc.__('alkisSelect')}
            />
    }
}

class FormAuxButton extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);

        return <sidebar.AuxButton
            {...gws.lib.cls('alkisFormAuxButton', this.props.alkisTab === 'form' && 'isActive')}
            whenTouched={() => cc.goTo('form')}
            tooltip={cc.__('alkisGotoForm')}
        />
    }
}

class ListAuxButton extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);

        return <sidebar.AuxButton
            {...gws.lib.cls('alkisListAuxButton', this.props.alkisTab === 'list' && 'isActive')}
            whenTouched={() => cc.goTo('list')}
            tooltip={cc.__('alkisGotoList')}
        />
    }
}

class SelectionAuxButton extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);

        if (!cc.setup.ui.useSelect)
            return null;

        let sel = this.props.alkisFsSelection || [];

        return <sidebar.AuxButton
            {...gws.lib.cls('alkisSelectionAuxButton', this.props.alkisTab === 'selection' && 'isActive')}
            badge={sel.length ? String(sel.length) : null}
            whenTouched={() => cc.goTo('selection')}
            tooltip={cc.__('alkisGotoSelection')}
        />
    }
}

class ClearAuxButton extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);

        if (this.props.alkisFsSelection.length === 0)
            return null;

        return <sidebar.AuxButton
            className="alkisClearAuxButton"
            whenTouched={() => cc.clearSelection()}
            tooltip={cc.__('alkisClearSelection')}
        />
    }
}

class ResetAuxButton extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);

        return <sidebar.AuxButton
            className="alkisResetAuxButton"
            whenTouched={() => cc.reset()}
            tooltip={cc.__('alkisResetButton')}
        />
    }
}

class Navigation extends gws.View<ViewProps> {
    render() {
        return <React.Fragment>
            <FormAuxButton {...this.props}/>
            <ListAuxButton {...this.props}/>
            <SelectionAuxButton {...this.props}/>
        </React.Fragment>
    }
}

class LoaderTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={cc.__('alkisFormTitle')}/>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                <div className="alkisLoading">
                    {cc.__('alkisLoading')}
                    <gws.ui.Loader/>
                </div>
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

class MessageTab extends gws.View<MessageViewProps> {
    render() {
        let cc = _master(this);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={cc.__('alkisFormTitle')}/>
            </sidebar.TabHeader>

            <sidebar.EmptyTabBody>
                {this.props.error && <gws.ui.Error text={this.props.error}/>}
                {this.props.message && <gws.ui.Text content={this.props.message}/>}
                {this.props.withFormLink && <a onClick={() => cc.goTo('form')}>
                    {cc.__('alkisBackToForm')}
                </a>}
            </sidebar.EmptyTabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Navigation {...this.props}/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>

        </sidebar.Tab>
    }
}

class SearchForm extends gws.View<ViewProps> {

    render() {
        let cc = _master(this),
            setup = cc.setup,
            form = this.props.alkisFsFormValues;

        let boundTo = key => ({
            value: form[key],
            whenChanged: value => cc.updateForm({[key]: value}),
            whenEntered: () => cc.search()
        });

        let nameShowMode = '';

        if (setup.withEigentuemer) {
            if (!setup.withEigentuemerControl)
                nameShowMode = 'enabled';
            else if (form.wantEigentuemer)
                nameShowMode = 'enabled';
            else
                nameShowMode = '';
        }

        let gemarkungListMode = cc.setup.ui.gemarkungListMode;

        let gemarkungListValue = '';

        if (form.gemarkungCode)
            gemarkungListValue = _PREFIX_GEMARKUNG + ':' + form.gemarkungCode;
        else if (form.gemeindeCode)
            gemarkungListValue = _PREFIX_GEMEINDE + ':' + form.gemeindeCode;

        let strasseSearchMode = {
            opts: cc.setup.strasseSearchOptions,
            extraText: 'separate',
        }

        return <Form>
            {nameShowMode && <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={cc.__('alkisVorname')}
                        disabled={nameShowMode === 'disabled'}
                        {...boundTo('personVorname')}
                        withClear
                    />
                </Cell>
            </Row>}

            {nameShowMode && <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={cc.__('alkisNachname')}
                        disabled={nameShowMode === 'disabled'}
                        {...boundTo('personName')}
                        withClear
                    />
                </Cell>
            </Row>}


            {gemarkungListMode !== gws.api.plugin.alkis.action.GemarkungListMode.none && <Row>
                <Cell flex>
                    <gws.ui.Select
                        placeholder={
                            gemarkungListMode === gws.api.plugin.alkis.action.GemarkungListMode.tree
                                ? cc.__('alkisGemeindeGemarkung')
                                : cc.__('alkisGemarkung')
                        }
                        items={this.props.alkisFsGemarkungListItems}
                        value={gemarkungListValue}
                        whenChanged={value => cc.whenGemarkungChanged(value)}
                        withSearch
                        withClear
                    />
                </Cell>
            </Row>}

            <Row>
                <Cell flex>
                    <gws.ui.Select
                        placeholder={cc.__('alkisStrasse')}
                        items={this.props.alkisFsStrasseListItems}
                        {...boundTo('strasseCode')}
                        searchMode={strasseSearchMode}
                        withSearch
                        withClear
                    />
                </Cell>
                <Cell width={90}>
                    <gws.ui.TextInput
                        placeholder={cc.__('alkisNr')}
                        {...boundTo('hausnummer')}
                        withClear
                    />
                </Cell>
            </Row>

            <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={
                            setup.withFlurnummer
                                ? cc.__('alkisVnumFlur')
                                : cc.__('alkisVnum')
                        }
                        {...boundTo('fsnummer')}
                        withClear
                    />
                </Cell>
            </Row>

            <Row>
                <Cell flex>
                    <gws.ui.NumberInput
                        placeholder={cc.__('alkisAreaFrom')}
                        {...boundTo('flaecheVon')}
                        withClear
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.NumberInput
                        placeholder={cc.__('alkisAreaTo')}
                        {...boundTo('flaecheBis')}
                        withClear
                    />
                </Cell>
            </Row>

            {setup.withBuchung && <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={cc.__('alkisBblatt')}
                        {...boundTo('bblatt')}
                        withClear
                    />
                </Cell>
            </Row>}

            {setup.withEigentuemerControl && <Row className='alkisControlToggle'>
                <Cell flex>
                    <gws.ui.Toggle
                        type="checkbox"
                        {...boundTo('wantEigentuemer')}
                        label={cc.__('alkisWantEigentuemer')}
                        inline={true}
                    />
                </Cell>
            </Row>}

            {setup.withEigentuemerControl && form.wantEigentuemer && <Row>
                <Cell flex>
                    <gws.ui.TextArea
                        {...boundTo('eigentuemerControlInput')}
                        placeholder={cc.__('alkisControlInput')}
                    />
                </Cell>
            </Row>}

            {setup.ui.useHistory && <Row>
                <Cell flex>
                    <gws.ui.Toggle
                        type="checkbox"
                        {...boundTo('wantHistorySearch')}
                        label={cc.__('alkisWantHistorySearch')}
                        inline={true}
                    />
                </Cell>
            </Row>}

            {setup.ui.useHistory && <Row>
                <Cell flex>
                    <gws.ui.Toggle
                        type="checkbox"
                        {...boundTo('wantHistoryDisplay')}
                        label={cc.__('alkisWantHistoryDisplay')}
                        inline={true}
                    />
                </Cell>
            </Row>}


            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.Button
                        {...gws.lib.cls('alkisSearchSubmitButton')}
                        tooltip={cc.__('alkisSubmitButton')}
                        whenTouched={() => cc.formSearch()}
                    />
                </Cell>
                {setup.ui.searchSelection && <Cell>
                    <gws.ui.Button
                        {...gws.lib.cls('alkisSearchSelectionButton')}
                        tooltip={cc.__('alkisSelectionSearchButton')}
                        whenTouched={() => cc.selectionSearch()}
                    />
                </Cell>}
                {setup.ui.searchSpatial && <Cell>
                    <gws.ui.Button
                        {...gws.lib.cls('alkisSearchLensButton', this.props.appActiveTool === 'Tool.Alkis.Lens' && 'isActive')}
                        tooltip={cc.__('alkisLensButton')}
                        whenTouched={() => cc.startLens()}
                    />
                </Cell>}
                {setup.ui.usePick && <Cell>
                    <gws.ui.Button
                        {...gws.lib.cls('alkisPickButton', this.props.appActiveTool === 'Tool.Alkis.Pick' && 'isActive')}
                        tooltip={cc.__('alkisPickButton')}
                        whenTouched={() => cc.startPick()}
                    />
                </Cell>}
                <Cell>
                    <gws.ui.Button
                        {...gws.lib.cls('alkisSearchResetButton')}
                        tooltip={cc.__('alkisResetButton')}
                        whenTouched={() => cc.reset()}
                    />
                </Cell>
            </Row>
        </Form>;
    }
}

class FormTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={cc.__('alkisFormTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <SearchForm {...this.props} />
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Navigation {...this.props}/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>


        </sidebar.Tab>
    }
}

class FeatureList extends gws.View<ViewProps> {

    render() {
        let cc = _master(this);

        let rightButton = f => cc.isSelected(f)
            ? <components.list.Button
                className="alkisUnselectListButton"
                whenTouched={() => cc.unselect([f])}
            />

            : <components.list.Button
                className="alkisSelectListButton"
                whenTouched={() => cc.select([f])}
            />
        ;

        if (!cc.setup.ui.useSelect)
            rightButton = null;

        let content = f => <gws.ui.Link
            whenTouched={() => cc.showDetails(f)}
            content={f.views.teaser}
        />;

        return <components.feature.List
            controller={cc}
            features={this.props.features}
            content={content}
            isSelected={f => this.props.showSelection && cc.isSelected(f)}
            rightButton={rightButton}
            withZoom
        />

    }

}

class ListTab extends gws.View<ViewProps> {
    title() {
        let cc = _master(this);

        let total = this.props.alkisFsResultCount,
            disp = this.props.alkisFsResults.length;

        if (!disp)
            return cc.__('alkisFormTitle');

        let s = (total > disp)
            ? cc.__('alkisSearchResultsPartial')
            : cc.__('alkisSearchResults');

        s = s.replace(/\$1/g, String(disp));
        s = s.replace(/\$2/g, String(total));

        return s;
    }

    render() {
        let cc = _master(this);
        let features = this.props.alkisFsResults;

        if (gws.lib.isEmpty(features)) {
            return <MessageTab
                {...this.props}
                message={cc.__('alkisNotFound')}
                withFormLink={true}
            />;
        }

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.title()}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <FeatureList {...this.props} features={features} showSelection={true}/>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Navigation {...this.props}/>
                    <Cell flex/>
                    <SelectAuxButton {...this.props} features={features}/>
                    <ExportAuxButton {...this.props} features={features}/>
                    <PrintAuxButton {...this.props} features={features}/>
                    <ResetAuxButton {...this.props} features={features}/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class SelectionTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);
        let features = this.props.alkisFsSelection;
        let hasFeatures = !gws.lib.isEmpty(features);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={cc.__('alkisSelectionTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {hasFeatures
                    ? <FeatureList {...this.props} features={features} showSelection={false}/>
                    : <sidebar.EmptyTabBody>{cc.__('alkisNoSelection')}</sidebar.EmptyTabBody>
                }
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Navigation {...this.props}/>
                    <Cell flex/>
                    {hasFeatures && <ExportAuxButton {...this.props} features={features}/>}
                    {hasFeatures && <PrintAuxButton {...this.props} features={features}/>}
                    <storage.AuxButtons
                        controller={cc}
                        actionName="alkisSelectionStorage"
                        hasData={hasFeatures}
                        getData={() => cc.storageGetData()}
                        loadData={(data) => cc.storageLoadData(data)}
                    />
                    {hasFeatures && <ClearAuxButton {...this.props} />}
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class DetailsTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);
        let feature = this.props.alkisFsDetailsFeature;

        return <sidebar.Tab>

            <sidebar.TabHeader>
                <gws.ui.Title content={cc.__('alkisInfoTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <components.Description content={feature.views.description}/>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Navigation {...this.props}/>
                    <Cell flex/>
                    <ToggleAuxButton {...this.props} features={[feature]}/>
                    <PrintAuxButton {...this.props} features={[feature]}/>
                    <Cell>
                        <components.feature.TaskButton
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

class ExportTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);

        let allGroups = cc.setup.exportGroups,
            selectedGroupIndexes = this.props.alkisFsExportGroupIndexes;

        let changed = (gid, value) => cc.update({
            alkisFsExportGroupIndexes: selectedGroupIndexes.filter(g => g !== gid).concat(value ? [gid] : [])
        });

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={cc.__('alkisExportTitle')}/>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                <div className="alkisFsDetailsTabContent">
                    <Form>
                        <Row>
                            <Cell flex>
                                <gws.ui.Group vertical>
                                    {allGroups.map(g => <gws.ui.Toggle
                                            key={g.index}
                                            type="checkbox"
                                            inline
                                            label={g.title}
                                            value={selectedGroupIndexes.includes(g.index)}
                                            whenChanged={value => changed(g.index, value)}
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
                                    whenTouched={() => cc.submitExport()}
                                    label={cc.__('alkisExportButton')}
                                />
                            </Cell>
                        </Row>
                    </Form>
                </div>
            </sidebar.TabBody>
            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Navigation {...this.props}/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>

    }
}

class SidebarView extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);

        if (!cc.setup)
            return <sidebar.EmptyTab>{cc.__('alkisNoData')}</sidebar.EmptyTab>;

        if (this.props.alkisFsLoading)
            return <LoaderTab {...this.props}/>;

        let tab = this.props.alkisTab;

        if (!tab || tab === 'form')
            return <FormTab {...this.props} />;

        if (tab === 'list')
            return <ListTab {...this.props} />;

        if (tab === 'details')
            return <DetailsTab {...this.props} />;

        if (tab === 'error')
            return <MessageTab {...this.props} error={this.props.alkisFsError} withFormLink={true}/>;

        if (tab === 'selection')
            return <SelectionTab {...this.props} />;

        if (tab === 'export')
            return <ExportTab {...this.props} />;
    }
}

class Sidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'alkisSidebarIcon';

    get tooltip() {
        return this.__('alkisSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarView, StoreKeys));
    }

}

class LensTool extends lens.Tool {
    title = this.__('alkisLensTitle');

    async runSearch(geom) {
        let cc = _master(this);

        cc.updateForm({shapes: [this.map.geom2shape(geom)]});
        await cc.search();
    }

    stop() {
        let cc = _master(this);

        super.stop();
        cc.updateForm({shapes: null});
    }
}

class PickTool extends gws.Tool {
    start() {
        let cc = _master(this);

        this.map.prependInteractions([
            this.map.pointerInteraction({
                whenTouched: evt => cc.pickTouched(evt.coordinate),
            }),
        ]);
    }

    stop() {

    }
}

class Controller extends gws.Controller {
    uid = MASTER;
    history: Array<string>;
    selectionLayer: gws.types.IFeatureLayer;
    setup: gws.api.plugin.alkis.action.Props;
    toponyms: Toponyms;

    async init() {
        this.setup = this.app.actionProps('alkis');
        if (!this.setup)
            return;

        this.history = [];

        console.time('ALKIS:toponyms:load');

        this.toponyms = await this.loadToponyms();

        console.timeEnd('ALKIS:toponyms:load');

        this.updateObject('storageState', {
            alkisSelectionStorage: this.setup.storage ? this.setup.storage.state : null,
        });

        this.update({
            alkisTab: 'form',
            alkisFsLoading: false,

            alkisFsFormValues: {},
            alkisFsExportGroupIndexes: [],

            alkisFsGemarkungListItems: this.gemarkungListItems(),
            alkisFsStrasseListItems: this.strasseListItems(this.toponyms.strassen),

            alkisFsSearchResponse: null,
            alkisFsDetailsResponse: null,
            alkisFsSelection: [],
        });

        this.app.whenLoaded(() => this.findFlurstueckFromUrl());
    }

    async findFlurstueckFromUrl() {
        let p, res = null;

        p = this.app.urlParams['alkisFs'];
        if (p) {
            res = await this.app.server.alkisFindFlurstueck({combinedFlurstueckCode: p});
        }

        p = this.app.urlParams['alkisAd'];
        if (p) {
            res = await this.app.server.alkisFindAdresse({combinedAdresseCode: p});
        }

        if (!res || res.error) {
            return;
        }

        let features = this.map.readFeatures(res.features);

        if (features.length > 0) {
            this.update({
                marker: {
                    features: [features[0]],
                    mode: 'draw zoom',
                },
                infoboxContent: <components.Infobox
                    controller={this}>{features[0].views.teaser}</components.Infobox>,
            });
        }

    }

    async loadToponyms(): Promise<Toponyms> {
        let res = await this.app.server.alkisGetToponyms({});

        let t: Toponyms = {
            gemeinden: [],
            gemarkungen: [],
            gemeindeIndex: {},
            gemarkungIndex: {},
            strassen: []
        };

        for (let g of res.gemeinde) {
            let d = {name: g[0], code: g[1]};
            t.gemeinden.push(d);
            t.gemeindeIndex[g[1]] = d;
        }
        for (let g of res.gemarkung) {
            let d = {name: g[0], code: g[1], gemeindeCode: g[2]};
            t.gemarkungen.push(d);
            t.gemarkungIndex[g[1]] = d;
        }
        let n = 0;
        for (let s of res.strasse) {
            t.strassen.push({name: s[0], code: String(n), gemeindeCode: s[1], gemarkungCode: s[2]});
            n += 1;
        }
        return t;
    }

    gemarkungListItems(): Array<gws.ui.ListItem> {
        let items = [];

        switch (this.setup.ui.gemarkungListMode) {

            case gws.api.plugin.alkis.action.GemarkungListMode.plain:
                for (let g of this.toponyms.gemarkungen) {
                    items.push({
                        text: g.name,
                        value: _PREFIX_GEMARKUNG + ':' + g.code,
                    })
                }
                break;

            case gws.api.plugin.alkis.action.GemarkungListMode.combined:
                for (let g of this.toponyms.gemarkungen) {
                    items.push({
                        text: g.name,
                        extraText: this.toponyms.gemeindeIndex[g.gemeindeCode].name,
                        value: _PREFIX_GEMARKUNG + ':' + g.code,
                    });
                }
                break;

            case gws.api.plugin.alkis.action.GemarkungListMode.tree:
                for (let gd of this.toponyms.gemeinden) {
                    items.push({
                        text: gd.name,
                        value: _PREFIX_GEMEINDE + ':' + gd.code,
                        level: 1,
                    });
                    for (let gk of this.toponyms.gemarkungen) {
                        if (gk.gemeindeCode === gd.code) {
                            items.push({
                                text: gk.name,
                                value: _PREFIX_GEMARKUNG + ':' + gk.code,
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
        let strasseCounts = {}, ls = [];

        for (let s of strassen) {
            strasseCounts[s.name] = (strasseCounts[s.name] || 0) + 1;
        }

        switch (this.setup.ui.strasseListMode) {

            case gws.api.plugin.alkis.action.StrasseListMode.plain:
                ls = strassen.map(s => ({
                    text: s.name,
                    value: s.code,
                    extraText: '',
                }));
                break;

            case gws.api.plugin.alkis.action.StrasseListMode.withGemarkung:
                ls = strassen.map(s => ({
                    text: s.name,
                    value: s.code,
                    extraText: this.toponyms.gemarkungIndex[s.gemarkungCode].name,
                }));
                break;

            case gws.api.plugin.alkis.action.StrasseListMode.withGemarkungIfRepeated:
                ls = strassen.map(s => ({
                    text: s.name,
                    value: s.code,
                    extraText: strasseCounts[s.name] > 1
                        ? this.toponyms.gemarkungIndex[s.gemarkungCode].name
                        : ''
                }));
                break;

            case gws.api.plugin.alkis.action.StrasseListMode.withGemeinde:
                ls = strassen.map(s => ({
                    text: s.name,
                    value: s.code,
                    extraText: this.toponyms.gemeindeIndex[s.gemeindeCode].name,
                }));
                break;

            case gws.api.plugin.alkis.action.StrasseListMode.withGemeindeIfRepeated:
                ls = strassen.map(s => ({
                    text: s.name,
                    value: s.code,
                    extraText: strasseCounts[s.name] > 1
                        ? this.toponyms.gemeindeIndex[s.gemeindeCode].name
                        : ''
                }));
                break;
        }

        return ls.sort((a, b) => a.text.localeCompare(b.text) || a.extraText.localeCompare(b.extraText));
    }

    async whenGemarkungChanged(value) {
        let strassen = this.toponyms.strassen,
            form: FormValues = {
                gemarkungCode: null,
                gemeindeCode: null,
                strasseCode: null,
            };

        if (value) {
            let p = value.split(':');

            if (p[0] === _PREFIX_GEMEINDE) {
                form.gemeindeCode = p[1];
                strassen = this.toponyms.strassen.filter(s => s.gemeindeCode === form.gemeindeCode);
            }
            if (p[0] === _PREFIX_GEMARKUNG) {
                form.gemarkungCode = p[1];
                form.gemeindeCode = this.toponyms.gemarkungIndex[p[1]].gemeindeCode;
                strassen = this.toponyms.strassen.filter(s => s.gemarkungCode === form.gemarkungCode);
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
            let gs = gws.lib.compact(m.features.map((f: gws.types.IFeature) => f.geometry));
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

        let res = await this.app.server.alkisFindFlurstueck({
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

        let res = await this.app.server.alkisFindFlurstueck(this.fsSearchRequest());

        if (res.error) {
            return this.showError(res)
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

    fsSearchRequest(): gws.api.plugin.alkis.action.FindFlurstueckRequest {
        let req = {...this.getValue('alkisFsFormValues')};

        if (!gws.lib.isEmpty(req.strasseCode)) {
            let strasse = this.toponyms.strassen[Number(req.strasseCode)];
            if (strasse) {
                req.gemarkungCode = strasse.gemarkungCode;
                req.strasse = strasse.name;
            }
        }

        delete req.strasseCode;
        return req;
    }


    fsDetailsRequest(fs: Array<gws.types.IFeature>): gws.api.plugin.alkis.action.FindFlurstueckRequest {
        let form = this.getValue('alkisFsFormValues');

        let displayThemes = [
            gws.api.plugin.alkis.data.types.DisplayTheme.lage,
            gws.api.plugin.alkis.data.types.DisplayTheme.gebaeude,
            gws.api.plugin.alkis.data.types.DisplayTheme.nutzung,
        ]

        if (this.setup.withBuchung)
            displayThemes.push(gws.api.plugin.alkis.data.types.DisplayTheme.buchung)

        if (this.setup.withEigentuemer && (!this.setup.withEigentuemerControl || form.wantEigentuemer))
            displayThemes.push(gws.api.plugin.alkis.data.types.DisplayTheme.eigentuemer)

        return {
            displayThemes,
            eigentuemerControlInput: form.eigentuemerControlInput,
            uids: fs.map(f => f.uid),
            wantHistorySearch: form.wantHistorySearch,
            wantHistoryDisplay: form.wantHistoryDisplay,
        }
    }

    async showDetails(f: gws.types.IFeature, highlight = true) {
        let res = await this.app.server.alkisFindFlurstueck(this.fsDetailsRequest([f]));
        if (res.error) {
            return this.showError(res);
        }

        let features = this.map.readFeatures(res.features);

        if (features.length > 0) {
            if (highlight)
                this.highlight(features[0]);

            this.update({
                alkisFsDetailsFeature: features[0],
            });

            this.goTo('details');
        }
    }

    async startPrint(fs: Array<gws.types.IFeature>) {

        this.update({
            printJob: {state: gws.api.core.JobState.init},
            marker: null,
        });

        let level = this.setup.printer.qualityLevels[0];
        let dpi = level ? level.dpi : 0;
        let featureStyle = this.app.style.get('.alkisFeature').props;

        let mapParams = await this.map.printParams(null, dpi);
        mapParams.planes = mapParams.planes.filter(p => p.layerUid !== '_alkisSelectLayer');

        let printRequest: gws.api.core.PrintRequest = {
            type: gws.api.core.PrintRequestType.template,
            printerUid: this.setup.printer.uid,
            dpi,
            maps: [mapParams]
        };

        let q = {
            findRequest: this.fsDetailsRequest(fs),
            printRequest,
            featureStyle,
        };

        this.update({
            printerJob: await this.app.server.alkisPrintFlurstueck(q),
            printerSnapshotMode: false,
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
                uid: '_alkisSelectLayer',
                cssSelector: '.alkisSelectFeature',
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

    storageGetData() {
        let fs = this.getValue('alkisFsSelection');
        return {
            selection: fs.map(f => f.uid)
        }
    }

    async storageLoadData(data) {
        this.clearSelection();

        if (!data || !data.selection)
            return;

        let res = await this.app.server.alkisFindFlurstueck({
            uids: data.selection,
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
            alkisFsStrasseListItems: [],
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

    async submitExport() {
        let fs: Array<gws.types.IFeature> = this.getValue('alkisFsExportFeatures');

        let q = {
            findRequest: this.fsDetailsRequest(fs),
            groupIndexes: this.getValue('alkisFsExportGroupIndexes'),
        };

        let res = await this.app.server.alkisExportFlurstueck(q, {binaryResponse: true});
        if (res.error) {
            return;
        }
        gws.lib.downloadContent(res.content, res.mime, EXPORT_PATH)
    }

    showError(res) {
        let msg = this.__('alkisErrorGeneric');

        if (res.status === 403) {
            msg = this.__('alkisErrorForbidden');
        }

        if (res.status === 409) {
            msg = this.__('alkisErrorTooMany').replace(/\$1/g, this.setup.limit);
        }

        this.update({
            alkisFsError: msg,
        });

        this.update({alkisFsLoading: false});
        return this.goTo('error');
    }

    goTo(tab) {
        if (this.history[this.history.length - 1] !== tab)
            this.history.push(tab);
        this.update({
            alkisTab: tab
        });
        if (tab === 'form') {
            this.updateForm({eigentuemerControlInput: null});
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

gws.registerTags({
    [MASTER]: Controller,
    'Sidebar.Alkis': Sidebar,
    'Tool.Alkis.Lens': LensTool,
    'Tool.Alkis.Pick': PickTool,
});
