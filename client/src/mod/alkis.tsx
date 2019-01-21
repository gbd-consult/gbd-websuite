import * as React from 'react';
import * as gws from 'gws';

import * as sidebar from './common/sidebar';
import * as lens from './common/lens';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Alkis';


function _master(view) {
    return view.props.controller.app.controller(MASTER) as AlkisController;
}

const STRINGS = {

    formTitle: 'Flurstückssuche',
    infoTitle: "Informationen",
    selectionTitle: "Ablage",
    exportTitle: "Export",

    gotoForm: 'Suche',
    gotoList: 'Ergebnisse',
    gotoSelection: 'Ablage',

    print: 'Ausdrucken',
    highlight: 'Auf der Karte zeigen',
    selectAll: 'Alle auswählen',
    unselect: 'Abwählen',
    select: 'Auswählen',
    clearSelection: 'Ablage leeren',
    goBack: 'Zurück',

    loading: 'Daten werden geladen...',
    backToForm: 'Zurück zum Formular',

    noData: 'Flurstücksinformationen konnten nicht gefunden werden',
    notFound: 'Zu Ihrer Suche wurden keine Ergebnisse gefunden.',
    noSelection: 'Die Ablage ist leer',

    errorGeneric: 'Es ist ein Fehler aufgetreten.',
    errorControl: 'Validierung fehlgeschlagen.  Bitte überprüfen Sie Ihre Angaben.',

    vorname: "Vorname",
    nachname: "Nachname",
    gemarkung: "Gemarkung",
    strasse: "Straße",
    nr: "Nr",
    vnum: "Zähler/Nenner",
    vnumFlur: "Flur-Zähler/Nenner",
    areaFrom: "Fläche von m\u00b2",
    areaTo: "bis m\u00b2",
    bblatt: 'Buchungsblattnummer',
    wantEigentuemer: "Zugang zu Personendaten",
    controlInput: 'Abrufgrund',

    submitButton: "Suchen",
    lensButton: "Räumliche Suche",
    resetButton: "Neu",
    exportButton: "Exportieren",

    searchResults2: "Suchergebnisse ($1 von $2)",
    searchResults: "Suchergebnisse ($1)",

    exportPath: 'fsinfo.csv',

};

const exportGroups = [
    ['base', 'Basisdaten'],
    ['lage', 'Lage'],
    ['gebaeude', 'Gebäude'],
    ['buchung', 'Buchungsblatt'],
    ['eigentuemer', 'Eigentümer'],
    ['nutzung', 'Nutzung'],
];

type AlkisFsTab = 'form' | 'list' | 'details' | 'error' | 'selection' | 'export';

interface FsSearchProps extends gws.types.ViewProps {
    controller: AlkisController;

    alkisFsTab: AlkisFsTab;

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

    features?: Array<gws.types.IMapFeature>;
    appActiveTool: string;

}

const FsSearchStoreKeys = [
    'alkisFsTab',
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

class ExportCell extends gws.View<FsSearchProps> {
    render() {
        if (!this.props.alkisFsSetup.withExport || this.props.features.length === 0)
            return null;
        return <Cell>
            <gws.ui.IconButton
                className="modAlkisExportButton"
                whenTouched={() => _master(this).startExport(this.props.features)}
                tooltip={STRINGS.exportTitle}
            />
        </Cell>;
    }
}

class PrintCell extends gws.View<FsSearchProps> {
    render() {
        if (!this.props.alkisFsSetup.printTemplate || this.props.features.length === 0)
            return null;
        return <Cell>
            <gws.ui.IconButton
                className="modAlkisPrintButton"
                whenTouched={() => _master(this).startPrint(this.props.features)}
                tooltip={STRINGS.print}

            />
        </Cell>;
    }
}

class HighlightCell extends gws.View<FsSearchProps> {
    render() {
        if (!this.props.alkisFsSetup.printTemplate || this.props.features.length === 0)
            return null;
        return <Cell>
            <gws.ui.IconButton
                className="modAlkisHighlightButton"
                whenTouched={() => _master(this).highlightMany(this.props.features)}
                tooltip={STRINGS.highlight}

            />
        </Cell>;
    }
}

class SelectAllCell extends gws.View<FsSearchProps> {
    render() {
        if (!this.props.alkisFsSetup.withSelect)
            return null;
        return <Cell>
            <gws.ui.IconButton
                className="modAlkisSelectAllButton"
                whenTouched={() => _master(this).select(this.props.features)}
                tooltip={STRINGS.selectAll}

            />
        </Cell>
    }
}

class ToggleSelectionCell extends gws.View<FsSearchProps> {
    render() {
        if (!this.props.alkisFsSetup.withSelect)
            return null;

        let cc = _master(this),
            feature = this.props.features[0];

        if (cc.isSelected(feature))
            return <Cell>
                <gws.ui.IconButton
                    className="modAlkisUnselectButton"
                    whenTouched={() => _master(this).unselect([feature])}
                    tooltip={STRINGS.unselect}
                />
            </Cell>;
        else
            return <Cell>
                <gws.ui.IconButton
                    className="modAlkisSelectButton"
                    whenTouched={() => _master(this).select([feature])}
                    tooltip={STRINGS.select}
                />
            </Cell>;
    }
}

class GotoFormCell extends gws.View<FsSearchProps> {
    render() {
        return <Cell>
            <gws.ui.IconButton
                {...gws.tools.cls('modAlkisGotoFormButton', this.props.alkisFsTab === 'form' && 'isActive')}
                whenTouched={() => _master(this).goTo('form')}
                tooltip={STRINGS.gotoForm}
            />
        </Cell>
    }
}

class GotoListCell extends gws.View<FsSearchProps> {
    render() {
        return <Cell>
            <gws.ui.IconButton
                {...gws.tools.cls('modAlkisGotoListButton', this.props.alkisFsTab === 'list' && 'isActive')}
                whenTouched={() => _master(this).goTo('list')}
                tooltip={STRINGS.gotoList}
            />
        </Cell>
    }
}

class GotoSelectionCell extends gws.View<FsSearchProps> {
    render() {
        if (!this.props.alkisFsSetup.withSelect)
            return null;

        let sel = this.props.alkisFsSelection || [];

        return <Cell>
            <gws.ui.IconButton
                {...gws.tools.cls('modAlkisGotoSelectionButton', this.props.alkisFsTab === 'selection' && 'isActive')}
                badge={sel.length ? String(sel.length) : null}
                whenTouched={() => _master(this).goTo('selection')}
                tooltip={STRINGS.gotoSelection}
            />
        </Cell>
    }
}

class Navigation extends gws.View<FsSearchProps> {
    render() {
        return <React.Fragment>
            <GotoFormCell {...this.props}/>
            <GotoListCell {...this.props}/>
            <GotoSelectionCell {...this.props}/>
        </React.Fragment>
    }
}

class ClearSelectionCell extends gws.View<FsSearchProps> {
    render() {
        if (this.props.alkisFsSelection.length === 0)
            return null;

        return <Cell>
            <gws.ui.IconButton
                className="modAlkisClearSelectionButton"
                whenTouched={() => _master(this).clearSelection()}
                tooltip={STRINGS.clearSelection}
            />
        </Cell>
    }
}

class LoaderTab extends React.PureComponent<{}> {
    render() {
        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={STRINGS.formTitle}/>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                <div className="modAlkisLoading">
                    {STRINGS.loading}
                    <gws.ui.Loader/>
                </div>
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

class ErrorTab extends gws.View<FsSearchProps> {
    render() {
        let cc = _master(this);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={STRINGS.formTitle}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <div className='modAlkisEmptyTab'>
                    <gws.ui.Error text={this.props.alkisFsError}/>
                    <a onClick={() => cc.goTo('form')}>{STRINGS.backToForm}</a>
                </div>
            </sidebar.TabBody>
            <sidebar.TabFooter>
                <sidebar.SecondaryToolbar>
                    <Navigation {...this.props}/>
                    <Cell flex/>
                    <GotoSelectionCell {...this.props}/>
                </sidebar.SecondaryToolbar>
            </sidebar.TabFooter>

        </sidebar.Tab>
    }
}

class SearchForm extends gws.View<FsSearchProps> {

    render() {
        let cc = _master(this),
            setup = this.props.alkisFsSetup;

        let boundTo = param => ({
            value: this.props.alkisFsParams[param],
            whenChanged: value => cc.updateFsParams({[param]: value}),
            whenEntered: () => cc.search()
        });


        let nameShowMode = '';

        if (setup.withEigentuemer)
            if (!setup.withControl)
                nameShowMode = 'enabled';
            else if (this.props.alkisFsParams.wantEigentuemer)
                nameShowMode = 'enabled';
            else
                nameShowMode = 'disabled';

        return <Form>
            {nameShowMode && <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={STRINGS.vorname}
                        disabled={nameShowMode === 'disabled'}
                        {...boundTo('vorname')}
                        withClear
                    />
                </Cell>
            </Row>}
            {nameShowMode && <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={STRINGS.nachname}
                        disabled={nameShowMode === 'disabled'}
                        {...boundTo('name')}
                        withClear
                    />
                </Cell>
            </Row>}

            <Row>
                <Cell flex>
                    <gws.ui.Select
                        placeholder={STRINGS.gemarkung}
                        items={this.props.alkisFsGemarkungen}
                        value={this.props.alkisFsParams.gemarkungUid}
                        whenChanged={value => cc.whenGemarkungChanged(value)}
                        withSearch
                        withClear
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    <gws.ui.Select
                        placeholder={STRINGS.strasse}
                        items={this.props.alkisFsStrassen}
                        {...boundTo('strasse')}
                        withSearch
                        withClear
                    />
                </Cell>
                <Cell width={90}>
                    <gws.ui.TextInput
                        placeholder={STRINGS.nr}
                        {...boundTo('hausnummer')}
                        withClear
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={setup.withFlurnummer ? STRINGS.vnumFlur : STRINGS.vnum}
                        {...boundTo('vnum')}
                        withClear
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={STRINGS.areaFrom}
                        {...boundTo('flaecheVon')}
                        withClear
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={STRINGS.areaTo}
                        {...boundTo('flaecheBis')}
                        withClear
                    />
                </Cell>
            </Row>

            {setup.withBuchung && <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={STRINGS.bblatt}
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
                        label={STRINGS.wantEigentuemer}
                    />
                </Cell>
            </Row>}
            {setup.withControl && this.props.alkisFsParams.wantEigentuemer && <Row>
                <Cell flex>
                    <gws.ui.TextArea
                        {...boundTo('controlInput')}
                        placeholder={STRINGS.controlInput}
                    />
                </Cell>
            </Row>}
            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.IconButton
                        {...gws.tools.cls('modAlkisSubmitButton', this.props.appActiveTool !== 'Tool.Alkis.Lens' && 'isActive')}
                        tooltip={STRINGS.submitButton}
                        whenTouched={() => {
                            cc.stopLens();
                            cc.search();
                            }}
                    />
                </Cell>
                <Cell>
                    <gws.ui.IconButton
                        {...gws.tools.cls('modAlkisLensButton', this.props.appActiveTool === 'Tool.Alkis.Lens' && 'isActive')}
                        tooltip={STRINGS.lensButton}
                        whenTouched={() => cc.startLens()}
                    />
                </Cell>
                <Cell>
                    <gws.ui.IconButton
                        className="cmpButtonFormCancel"
                        tooltip={STRINGS.resetButton}
                        whenTouched={() => cc.reset()}
                    />
                </Cell>
            </Row>
        </Form>;
    }
}

class FormTab extends gws.View<FsSearchProps> {
    render() {
        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={STRINGS.formTitle}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <SearchForm {...this.props} />
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.SecondaryToolbar>
                    <Navigation {...this.props}/>
                </sidebar.SecondaryToolbar>
            </sidebar.TabFooter>


        </sidebar.Tab>
    }
}

class ListTab extends gws.View<FsSearchProps> {
    title() {
        let total = this.props.alkisFsResultCount,
            disp = this.props.alkisFsResults.length;

        if (!disp)
            return STRINGS.formTitle;

        let s = (total > disp) ? STRINGS.searchResults2 : STRINGS.searchResults;

        s = s.replace(/\$1/g, String(disp));
        s = s.replace(/\$2/g, String(total));

        return s;
    }

    render() {
        let cc = _master(this);
        let features = this.props.alkisFsResults;

        if (gws.tools.empty(features)) {

            return <sidebar.EmptyTab>
                {STRINGS.notFound}
                <a onClick={() => cc.goTo('form')}>{STRINGS.backToForm}</a>
            </sidebar.EmptyTab>
        }

        let leftIcon = f => <gws.ui.IconButton
            className="cmpFeatureZoomIcon"
            whenTouched={() => cc.highlight(f)}
        />;

        let rightIcon = f => cc.isSelected(f)
            ? <gws.ui.IconButton
                className="modAlkisFeatureUnselectIcon"
                whenTouched={() => cc.unselect([f])}
            />

            : <gws.ui.IconButton
                className="modAlkisFeatureSelectIcon"
                whenTouched={() => cc.select([f])}
            />
        ;

        if (!this.props.alkisFsSetup.withSelect)
            rightIcon = null;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.title()}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.components.feature.List
                    controller={cc}
                    features={features}
                    item={f => <gws.ui.Link
                        whenTouched={() => cc.showDetails(f)}
                        content={f.props.teaser}
                    />}
                    isSelected={f => cc.isSelected(f)}
                    leftIcon={leftIcon}
                    rightIcon={rightIcon}
                />
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.SecondaryToolbar>
                    <Navigation {...this.props}/>
                    <Cell flex/>
                    <HighlightCell {...this.props} features={features}/>
                    <SelectAllCell {...this.props} features={features}/>
                    <ExportCell {...this.props} features={features}/>
                    <PrintCell {...this.props} features={features}/>
                </sidebar.SecondaryToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class SelectionTab extends gws.View<FsSearchProps> {
    render() {
        let cc = _master(this),
            features = this.props.alkisFsSelection;

        if (gws.tools.empty(features)) {
            return <sidebar.EmptyTab>
                {STRINGS.noSelection}
                <a onClick={() => cc.goTo('form')}>{STRINGS.backToForm}</a>
            </sidebar.EmptyTab>;

        }

        let leftIcon = f => <gws.ui.IconButton
            className="cmpFeatureZoomIcon"
            whenTouched={() => cc.highlight(f)}
        />;

        let rightIcon = f => <gws.ui.IconButton
            className="modAlkisFeatureUnselectIcon"
            whenTouched={() => cc.unselect([f])}
        />;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={STRINGS.selectionTitle}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.components.feature.List
                    controller={cc}
                    features={features}
                    item={f => <gws.ui.Link
                        whenTouched={() => cc.showDetails(f)}
                        content={f.props.teaser}
                    />}

                    leftIcon={leftIcon}
                    rightIcon={rightIcon}
                />
            </sidebar.TabBody>
            <sidebar.TabFooter>
                <sidebar.SecondaryToolbar>
                    <Navigation {...this.props}/>
                    <Cell flex/>
                    <HighlightCell {...this.props} features={features}/>
                    <ExportCell {...this.props} features={features}/>
                    <PrintCell {...this.props} features={features}/>
                    <ClearSelectionCell {...this.props} />
                </sidebar.SecondaryToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class DetailsTab extends gws.View<FsSearchProps> {
    render() {
        let cc = _master(this);
        let feature = this.props.alkisFsDetailsFeature;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={STRINGS.infoTitle}/>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                <div className="modAlkisFsDetailsTabContent">
                    <gws.ui.TextBlock
                        className="cmpDescription"
                        content={feature.props.description}
                        withHTML
                    />
                </div>
            </sidebar.TabBody>
            <sidebar.TabFooter>
                <sidebar.SecondaryToolbar>
                    <Navigation {...this.props}/>
                    <Cell flex/>
                    <PrintCell {...this.props} features={[feature]}/>
                    <ToggleSelectionCell {...this.props} features={[feature]}/>
                </sidebar.SecondaryToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>

    }
}

class ExportTab extends gws.View<FsSearchProps> {
    render() {
        let groups = this.props.alkisFsExportGroups;

        let changed = (group, value) => _master(this).update({
            alkisFsExportGroups: groups.filter(g => g !== group).concat(value ? [group] : [])
        });

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={STRINGS.exportTitle}/>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                <div className="modAlkisFsDetailsTabContent">
                    <Form>
                        <Row>
                            <Cell flex>
                                {exportGroups.map(([group, name]) => {
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
                                >{STRINGS.exportButton}</gws.ui.TextButton>
                            </Cell>
                        </Row>
                    </Form>
                </div>
            </sidebar.TabBody>
            <sidebar.TabFooter>
                <sidebar.SecondaryToolbar>
                    <Navigation {...this.props}/>
                </sidebar.SecondaryToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>

    }
}

class SidebarTab extends gws.View<FsSearchProps> {
    render() {
        if (this.props.alkisFsLoading)
            return <LoaderTab/>;

        let tab = this.props.alkisFsTab;

        if (!tab || tab === 'form')
            return <FormTab {...this.props} />;

        if (tab === 'list')
            return <ListTab {...this.props} />;

        if (tab === 'details')
            return <DetailsTab {...this.props} />;

        if (tab === 'error')
            return <ErrorTab {...this.props} />;

        if (tab === 'selection')
            return <SelectionTab {...this.props} />;

        if (tab === 'export')
            return <ExportTab {...this.props} />;
    }
}

class AlkisLensTool extends lens.Tool {
    get master() {
        return this.app.controller(MASTER) as AlkisController;

    }

    async whenChanged(geom) {
        console.log('MASTER', this.master)
        this.master.updateFsParams({shape: this.map.geom2shape(geom)});
        await this.master.search();
    }

    stop() {
        super.stop();
        this.master.updateFsParams({shape: null});
    }
}

class AlkisSidebarController extends gws.Controller implements gws.types.ISidebarItem {
    get iconClass() {
        return 'modAlkisSidebarIcon';
    }

    get tooltip() {
        return this.__('modAlkisTooltip');
    }

    get tabView() {
        let master = this.app.controller(MASTER) as AlkisController;
        if (!master.setup)
            return <sidebar.EmptyTab>
                {STRINGS.noData}
            </sidebar.EmptyTab>;

        return this.createElement(
            this.connect(SidebarTab, FsSearchStoreKeys));
    }

}

class AlkisController extends gws.Controller {
    uid = MASTER;
    setup: gws.api.AlkisFsSetupResponse;
    history: Array<string>;

    updateFsParams(obj) {
        this.update({
            alkisFsParams: {
                ...this.getValue('alkisFsParams'),
                ...obj
            }
        });
    }

    async init() {
        let res = await this.app.server.alkisFsSetup({
            projectUid: this.app.project.uid
        });

        if (res.error) {
            this.setup = null;
            return;
        }

        await this.app.addTool('Tool.Alkis.Lens', this.app.createController(AlkisLensTool, this));


        this.setup = res;
        this.history = [];

        this.update({
            alkisFsSetup: this.setup,

            alkisFsTab: 'form',
            alkisFsLoading: false,

            alkisFsParams: {
                projectUid: this.app.project.uid
            },

            alkisFsExportGroups: ['base'],

            alkisFsStrassen: [],
            alkisFsGemarkungen: this.setup.gemarkungen.map(g => ({
                text: g.name,
                value: g.uid,
            })),

            alkisFsSearchResponse: null,
            alkisFsDetailsResponse: null,
            alkisFsSelection: [],
        });
    }

    async whenGemarkungChanged(value) {
        let strassen = [];

        if (value) {
            let res = await this.app.server.alkisFsStrassen({
                projectUid: this.app.project.uid,
                gemarkungUid: value,
            });

            if (!res.error)
                strassen = res.strassen.map(s => ({text: s, value: s}));
        }

        this.updateFsParams({
            gemarkungUid: value,
            strasse: ''
        });

        this.update({
            alkisFsStrassen: strassen
        });
    };

    startLens() {
        this.app.startTool('Tool.Alkis.Lens');
    }

    stopLens() {
        this.app.stopTool('Tool.Alkis.Lens');
    }

    async search() {

        this.update({alkisFsLoading: true});

        let res = await this.app.server.alkisFsSearch({
            ...this.getValue('alkisFsParams'),
            projectUid: this.app.project.uid
        });

        this.update({alkisFsLoading: false});

        if (res.error) {
            let msg = STRINGS.errorGeneric;

            if (res.error.status === 400) {
                msg = STRINGS.errorControl
            }

            this.update({
                alkisFsError: msg,
            });

            return this.goTo('error');
        }

        let features = this.map.readFeatures(res.features);

        this.update({
            alkisFsResults: features,
            alkisFsResultCount: res.total,
            marker: {
                features,
                mode: 'pan draw',
            }
        });

        this.goTo('list');
    }

    paramsForFeatures(fs: Array<gws.types.IMapFeature>) {
        let queryParams: gws.api.AlkisFsQueryParams = this.getValue('alkisFsParams');
        return {
            projectUid: this.app.project.uid,
            wantEigentuemer: queryParams.wantEigentuemer,
            controlInput: queryParams.controlInput,
            fsUids: fs.map(f => f.uid),
        }

    }

    async showDetails(f: gws.types.IMapFeature) {
        let q = this.paramsForFeatures([f]);
        let res = await this.app.server.alkisFsDetails(q);
        let feature = this.map.readFeature(res.feature);

        if (f) {
            this.highlight(f);

            this.update({
                alkisFsDetailsFeature: feature,
                marker: {
                    features: [feature],
                    mode: 'zoom draw'
                }
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
            highlightStyle: this.map.getStyleFromSelector('.modMarkerShape').props,
        };

        this.update({
            printJob: await this.app.server.alkisFsPrint(q)
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
        let sel = this.getValue('alkisFsSelection') || [],
            add = [];

        fs.forEach(f => {
            if (!featureIn(sel, f))
                add.push(f)
        });

        this.update({
            alkisFsSelection: sel.concat(add)
        });
    }

    unselect(fs: Array<gws.types.IMapFeature>) {
        let sel = this.getValue('alkisFsSelection') || [];

        this.update({
            alkisFsSelection: sel.filter(f => !featureIn(fs, f))
        });
    }

    clearSelection() {
        this.update({
            alkisFsSelection: []
        });
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
        });
        this.clearResults();
        this.stopLens();
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
            groups: exportGroups.map(grp => grp[0]).filter(g => groups.indexOf(g) >= 0),
        };
        let res = await this.app.server.alkisFsExport(q);

        let a = document.createElement('a');
        a.href = res.url;
        a.download = STRINGS.exportPath;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    goTo(tab) {
        if (this.history[this.history.length - 1] !== tab)
            this.history.push(tab);
        this.update({
            alkisFsTab: tab
        });
        if (tab === 'form') {
            this.updateFsParams({controlInput: ''});
        }
    }

}

export const tags = {
    [MASTER]: AlkisController,
    'Sidebar.Alkis': AlkisSidebarController,
};
