import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

import * as sidebar from './common/sidebar';
import * as lens from './common/lens';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Alkis';

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as AlkisController;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as AlkisController;
}

const STRINGS = {

    formTitle: 'Flurstückssuche',
    infoTitle: "Informationen",
    selectionTitle: "Ablage",
    exportTitle: "Export CSV",

    gotoForm: 'Suche',
    gotoList: 'Ergebnisse',
    gotoSelection: 'Ablage',

    print: 'Ducken',
    highlight: 'Auf der Karte zeigen',
    selectAll: 'Alle zur Ablage',
    unselect: 'Abwählen',
    select: 'Auswählen',
    clearSelection: 'Ablage leeren',
    goBack: 'Zurück',

    loadSelection: 'Ablage laden',
    saveSelection: 'Ablage speichern',

    loading: 'Daten werden geladen...',
    backToForm: 'Zurück zum Formular',

    noData: 'Flurstücksinformationen konnten nicht gefunden werden',
    notFound: 'Zu Ihrer Suche wurden keine Ergebnisse gefunden.',
    noSelection: 'Die Ablage ist leer',

    errorGeneric: 'Es ist ein Fehler aufgetreten.',
    errorControl: 'Validierung fehlgeschlagen. Bitte überprüfen Sie Ihre Angaben.',

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
    pickButton: "Flurstücke direkt auswählen",
    selectionSearchButton: "in der Auswahl Suchen",
    resetButton: "Neue Anfrage",
    exportButton: "Exportieren",

    searchResults2: "Suchergebnisse ($1 von $2)",
    searchResults: "Suchergebnisse ($1)",

    exportPath: 'fsinfo.csv',

};

const EXPORT_GROUPS = [
    ['base', 'Basisdaten'],
    ['lage', 'Lage'],
    ['gebaeude', 'Gebäude'],
    ['buchung', 'Buchungsblatt'],
    ['eigentuemer', 'Eigentümer'],
    ['nutzung', 'Nutzung'],
];

type AlkisAlkisTabName = 'form' | 'list' | 'details' | 'error' | 'selection' | 'export';

interface AlkisViewProps extends gws.types.ViewProps {
    controller: AlkisController;

    alkisTab: AlkisAlkisTabName;

    alkisFsSetup: gws.api.AlkisFsSetupResponse;

    alkisFsLoading: boolean;
    alkisFsError: string;
    alkisFsDialogMode: string;

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

    alkisFsSaveName: string;
    alkisFsSaveNames: Array<string>;

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
    'alkisFsDialogMode',
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
    'alkisFsSaveName',
    'alkisFsSaveNames',
    'appActiveTool',
];

function featureIn(fs: Array<gws.types.IMapFeature>, f: gws.types.IMapFeature) {
    return fs.some(g => g.uid === f.uid);
}

class AlkisExportAuxButton extends gws.View<AlkisViewProps> {
    render() {
        if (!this.props.alkisFsSetup.withExport || this.props.features.length === 0)
            return null;
        return <sidebar.AuxButton
            className="modAlkisExportAuxButton"
            whenTouched={() => _master(this).startExport(this.props.features)}
            tooltip={STRINGS.exportTitle}
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
            tooltip={STRINGS.print}
        />;
    }
}

class AlkisSaveAuxButton extends gws.View<AlkisViewProps> {
    render() {
        return <sidebar.AuxButton
            className="modAlkisSaveAuxButton"
            whenTouched={() => _master(this).saveSelection()}
            tooltip={STRINGS.saveSelection}
        />;
    }
}

class AlkisLoadAuxButton extends gws.View<AlkisViewProps> {
    render() {
        return <sidebar.AuxButton
            className="modAlkisLoadAuxButton"
            whenTouched={() => _master(this).loadSelection()}
            tooltip={STRINGS.loadSelection}
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
            tooltip={STRINGS.highlight}
        />
    }
}

class AlkisSelectAuxButton extends gws.View<AlkisViewProps> {
    render() {
        if (!this.props.alkisFsSetup.withSelect)
            return null;
        return <sidebar.AuxButton
            className="modAlkisSelectAuxButton"
            whenTouched={() => _master(this).select(this.props.features)}
            tooltip={STRINGS.selectAll}
        />
    }
}

class AlkisToggleAuxButton extends gws.View<AlkisViewProps> {
    render() {
        if (!this.props.alkisFsSetup.withSelect)
            return null;

        let master = _master(this),
            feature = this.props.features[0];

        if (master.isSelected(feature))
            return <sidebar.AuxButton
                className="modAlkisUnselectAuxButton"
                whenTouched={() => master.unselect([feature])}
                tooltip={STRINGS.unselect}
            />
        else
            return <sidebar.AuxButton
                className="modAlkisSelectAuxButton"
                whenTouched={() => master.select([feature])}
                tooltip={STRINGS.select}
            />
    }
}

class AlkisFormAuxButton extends gws.View<AlkisViewProps> {
    render() {
        return <sidebar.AuxButton
            {...gws.tools.cls('modAlkisFormAuxButton', this.props.alkisTab === 'form' && 'isActive')}
            whenTouched={() => _master(this).goTo('form')}
            tooltip={STRINGS.gotoForm}
        />
    }
}

class AlkisListAuxButton extends gws.View<AlkisViewProps> {
    render() {
        return <sidebar.AuxButton
            {...gws.tools.cls('modAlkisListAuxButton', this.props.alkisTab === 'list' && 'isActive')}
            whenTouched={() => _master(this).goTo('list')}
            tooltip={STRINGS.gotoList}
        />
    }
}

class AlkisSelectionAuxButton extends gws.View<AlkisViewProps> {
    render() {
        if (!this.props.alkisFsSetup.withSelect)
            return null;

        let sel = this.props.alkisFsSelection || [];

        return <sidebar.AuxButton
            {...gws.tools.cls('modAlkisSelectionAuxButton', this.props.alkisTab === 'selection' && 'isActive')}
            badge={sel.length ? String(sel.length) : null}
            whenTouched={() => _master(this).goTo('selection')}
            tooltip={STRINGS.gotoSelection}
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
            tooltip={STRINGS.clearSelection}
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

class AlkisLoaderTab extends React.PureComponent<{}> {
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

class AlkisMessageTab extends gws.View<AlkisMessageViewProps> {
    render() {
        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={STRINGS.formTitle}/>
            </sidebar.TabHeader>

            <sidebar.EmptyTabBody>
                {this.props.error && <gws.ui.Error text={this.props.error}/>}
                {this.props.message && <gws.ui.Text content={this.props.message}/>}
                {this.props.withFormLink && <a onClick={() => _master(this).goTo('form')}>{STRINGS.backToForm}</a>}
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
                        whenChanged={value => master.whenGemarkungChanged(value)}
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
                        {...gws.tools.cls('modAlkisSearchSubmitButton')}
                        tooltip={STRINGS.submitButton}
                        whenTouched={() => master.formSearch()}
                    />
                </Cell>
                <Cell>
                    <gws.ui.IconButton
                        {...gws.tools.cls('modAlkisSearchSelectionButton')}
                        tooltip={STRINGS.selectionSearchButton}
                        whenTouched={() => master.selectionSearch()}
                    />
                </Cell>
                <Cell>
                    <gws.ui.IconButton
                        {...gws.tools.cls('modAlkisSearchLensButton', this.props.appActiveTool === 'Tool.Alkis.Lens' && 'isActive')}
                        tooltip={STRINGS.lensButton}
                        whenTouched={() => master.startLens()}
                    />
                </Cell>
                <Cell>
                    <gws.ui.IconButton
                        {...gws.tools.cls('modAlkisPickButton', this.props.appActiveTool === 'Tool.Alkis.Pick' && 'isActive')}
                        tooltip={STRINGS.pickButton}
                        whenTouched={() => master.startPick()}
                    />
                </Cell>
                <Cell>
                    <gws.ui.IconButton
                        {...gws.tools.cls('modAlkisSearchCancelButton')}
                        tooltip={STRINGS.resetButton}
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
                <gws.ui.Title content={STRINGS.formTitle}/>
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

        if (!this.props.alkisFsSetup.withSelect)
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
            return STRINGS.formTitle;

        let s = (total > disp) ? STRINGS.searchResults2 : STRINGS.searchResults;

        s = s.replace(/\$1/g, String(disp));
        s = s.replace(/\$2/g, String(total));

        return s;
    }

    render() {
        let features = this.props.alkisFsResults;

        if (gws.tools.empty(features)) {
            return <AlkisMessageTab
                {...this.props}
                message={STRINGS.notFound}
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
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class AlkisSelectionTab extends gws.View<AlkisViewProps> {
    render() {
        let features = this.props.alkisFsSelection;
        let hasFeatures = !gws.tools.empty(features);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={STRINGS.selectionTitle}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {hasFeatures
                    ? <AlkisFeatureList {...this.props} features={features} showSelection={false}/>
                    : <sidebar.EmptyTabBody>{STRINGS.noSelection}</sidebar.EmptyTabBody>
                }
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <AlkisNavigation {...this.props}/>
                    <Cell flex/>
                    {hasFeatures && <AlkisSaveAuxButton {...this.props} features={features}/>}
                    <AlkisLoadAuxButton {...this.props} features={features}/>
                    {hasFeatures && <AlkisClearAuxButton {...this.props} />}
                    {hasFeatures && <AlkisExportAuxButton {...this.props} features={features}/>}
                    {hasFeatures && <AlkisPrintAuxButton {...this.props} features={features}/>}
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
                <gws.ui.Title content={STRINGS.infoTitle}/>
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
                <gws.ui.Title content={STRINGS.exportTitle}/>
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
                                >{STRINGS.exportButton}</gws.ui.TextButton>
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
            return <AlkisLoaderTab/>;

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
                {STRINGS.noData}
            </sidebar.EmptyTab>;

        return this.createElement(
            this.connect(AlkisSidebarView, AlkisStoreKeys));
    }

}

class AlkisLensTool extends lens.LensTool {
    get style() {
        return this.map.getStyleFromSelector('.modAlkisLensFeature');
    }

    async whenChanged(geom) {
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

class AlkisSelectDialog extends gws.View<AlkisViewProps> {

    render() {
        let mode = this.props.alkisFsDialogMode;

        if (!mode)
            return null;

        let cc = this.props.controller;

        let close = () => cc.update({alkisFsDialogMode: null});
        let update = v => cc.update({alkisFsSaveName: v});

        let title, submit, control;

        if (mode === 'save') {
            title = STRINGS.saveSelection;
            submit = () => cc.saveSelection2();
            control = <gws.ui.TextInput
                value={this.props.alkisFsSaveName}
                whenChanged={update}
                whenEntered={submit}
            />;
        }

        if (mode === 'load') {
            title = STRINGS.loadSelection;
            submit = () => cc.loadSelection2();
            control = <gws.ui.Select
                value={this.props.alkisFsSaveName}
                items={this.props.alkisFsSaveNames.map(s => ({
                    text: s,
                    value: s

                }))}
                whenChanged={update}
            />;
        }

        return <gws.ui.Dialog className="modAlkisSelectDialog" title={title} whenClosed={close}>
            <Form>
                <Row>
                    <Cell flex>{control}</Cell>
                </Row>
                <Row>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.IconButton className="cmpButtonFormOk" whenTouched={submit}/>
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton className="cmpButtonFormCancel" whenTouched={close}/>
                    </Cell>
                </Row>
            </Form>
        </gws.ui.Dialog>;
    }
}

class AlkisController extends gws.Controller {
    uid = MASTER;
    setup: gws.api.AlkisFsSetupResponse;
    history: Array<string>;
    selectionLayer: gws.types.IMapFeatureLayer;

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

        this.setup = res;
        this.history = [];

        this.update({
            alkisFsSetup: this.setup,

            alkisTab: 'form',
            alkisFsLoading: false,

            alkisFsParams: {},

            alkisFsExportGroups: ['base'],

            alkisFsStrassen: [],
            alkisFsGemarkungen: this.setup.gemarkungen.map(g => ({
                text: g.name,
                value: g.uid,
            })),

            alkisFsSearchResponse: null,
            alkisFsDetailsResponse: null,
            alkisFsSelection: [],

            alkisFsSaveName: '',
            alkisFsSaveNames: [],
        });

        await this.getSaveNames();

        this.app.whenLoaded(() => this.urlSearch());
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(AlkisSelectDialog, AlkisStoreKeys));
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
            projectUid: this.app.project.uid
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

    async urlSearch() {
        let p = this.app.urlParams['alkisFs'];

        if (p) {
            let res = await this.app.server.alkisFsSearch({
                projectUid: this.app.project.uid,
                alkisFs: p,
            });

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
            highlightStyle: this.map.getStyleFromSelector('.modMarkerFeature').props,
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
                mode: 'zoom draw fade'
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

        this.selectionLayer.addFeatures(this.getValue('alkisFsSelection'));
    }

    saveSelection() {
        this.update({
            alkisFsDialogMode: 'save'
        })
    }

    async saveSelection2() {
        let res = await this.app.server.alkisFsSaveSelection({
            projectUid: this.app.project.uid,
            name: this.getValue('alkisFsSaveName'),
            fsUids: this.getValue('alkisFsSelection').map(f => f.uid)
        });

        if (res) {
            this.update({
                alkisFsSaveNames: res.names
            });
        }

        this.update({
            alkisFsDialogMode: null,
        });
    }

    loadSelection() {
        this.update({
            alkisFsDialogMode: 'load'
        })
    }

    async loadSelection2() {
        let res = await this.app.server.alkisFsLoadSelection({
            projectUid: this.app.project.uid,
            name: this.getValue('alkisFsSaveName'),
        });

        if (res.features) {
            this.update({
                alkisFsSelection: this.map.readFeatures(res.features)
            });
        }

        this.update({
            alkisFsDialogMode: null,
        });
    }

    async getSaveNames() {
        let res = await this.app.server.alkisFsGetSaveNames({
            projectUid: this.app.project.uid,
        });
        this.update({
            alkisFsSaveNames: res.names || []
        })

    }

    clearSelection() {
        this.update({
            alkisFsSelection: []
        });
        this.map.removeLayer(this.selectionLayer)
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
        this.stopTools();
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
        a.download = STRINGS.exportPath;
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
