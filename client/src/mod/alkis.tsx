import * as React from 'react';
import * as gws from 'gws';

import * as sidebar from './sidebar';

let {Form, Row, Cell} = gws.ui.Layout;

const STRINGS = {

    formTitle: 'Flurstückssuche',
    infoTitle: "Informationen",
    selectionTitle: "Ablage",
    exportTitle: "Export",

    print: 'Ausdrucken',
    highlight: 'Auf der Karte zeigen',
    selectAll: 'Alle auswählen',
    unselect: 'Abwählen',
    select: 'Auswählen',
    gotoSelection: 'Ablage',
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
    vnum: "Kennzeichen",
    areaFrom: "Fläche von m\u00b2",
    areaTo: "bis m\u00b2",
    bblatt: 'Buchungsblattnummer',
    wantEigentuemer: "Zugang zu Personendaten",
    controlInput: 'Abrufgrund',

    submitButton: "Suchen",
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

interface FsSearchProps extends gws.types.ViewProps {
    controller: AlkisController;

    alkisFsTab: 'form' | 'list' | 'details' | 'error' | 'selection' | 'export'

    alkisFsSetup: gws.api.AlkisFsSetupResponse

    alkisFsLoading: boolean
    alkisFsError: string

    alkisFsExportGroups: Array<string>
    alkisFsExportFeatures: Array<gws.types.IMapFeature>

    alkisFsParams: gws.api.AlkisFsQueryParams

    alkisFsStrassen: Array<gws.ui.MenuItem>
    alkisFsGemarkungen: Array<gws.ui.MenuItem>

    alkisFsResults: Array<gws.types.IMapFeature>,
    alkisFsResultCount: number,

    alkisFsDetailsFeature: gws.types.IMapFeature,
    alkisFsDetailsText: string,

    alkisFsSelection: Array<gws.types.IMapFeature>,

    lensGeometryType: string;

}

interface FsSearchPropsWithFeatures extends FsSearchProps {
    features: Array<gws.types.IMapFeature>,
}

function featureIn(fs: Array<gws.types.IMapFeature>, f: gws.types.IMapFeature) {
    return fs.some(g => g.uid === f.uid);
}

class ExportCell extends gws.View<FsSearchPropsWithFeatures> {
    render() {
        if (!this.props.alkisFsSetup.withExport || this.props.features.length === 0)
            return null;
        return <Cell>
            <gws.ui.IconButton
                className="modAlkisExportButton"
                whenTouched={() => this.props.controller.startExport(this.props.features)}
                tooltip={STRINGS.exportTitle}
            />
        </Cell>;
    }
}

class PrintCell extends gws.View<FsSearchPropsWithFeatures> {
    render() {
        if (!this.props.alkisFsSetup.printTemplate || this.props.features.length === 0)
            return null;
        return <Cell>
            <gws.ui.PrintButton
                whenTouched={() => this.props.controller.startPrint(this.props.features)}
                tooltip={STRINGS.print}

            />
        </Cell>;
    }
}

class HighlightCell extends gws.View<FsSearchPropsWithFeatures> {
    render() {
        if (!this.props.alkisFsSetup.printTemplate || this.props.features.length === 0)
            return null;
        return <Cell>
            <gws.ui.IconButton
                className="modAlkisHighlightButton"
                whenTouched={() => this.props.controller.highlightMany(this.props.features)}
                tooltip={STRINGS.highlight}

            />
        </Cell>;
    }
}

class SelectAllCell extends gws.View<FsSearchPropsWithFeatures> {
    render() {
        if (!this.props.alkisFsSetup.withSelect)
            return null;
        return <Cell>
            <gws.ui.IconButton
                className="modAlkisSelectAllButton"
                whenTouched={() => this.props.controller.select(this.props.features)}
                tooltip={STRINGS.selectAll}

            />
        </Cell>
    }
}

class ToggleSelectionCell extends gws.View<FsSearchPropsWithFeatures> {
    render() {
        if (!this.props.alkisFsSetup.withSelect)
            return null;

        let cc = this.props.controller,
            feature = this.props.features[0];

        if (cc.isSelected(feature))
            return <Cell>
                <gws.ui.IconButton
                    className="modAlkisUnselectButton"
                    whenTouched={() => this.props.controller.unselect([feature])}
                    tooltip={STRINGS.unselect}
                />
            </Cell>;
        else
            return <Cell>
                <gws.ui.IconButton
                    className="modAlkisSelectButton"
                    whenTouched={() => this.props.controller.select([feature])}
                    tooltip={STRINGS.select}
                />
            </Cell>;
    }
}

class GotoSelectionCell extends gws.View<FsSearchProps> {
    render() {
        if (!this.props.alkisFsSetup.withSelect)
            return null;

        let sel = this.props.alkisFsSelection || [];

        if (!sel.length) {
            return null;
        }

        return <Cell>
            <gws.ui.IconButton
                className="modAlkisSelectionButton"
                badge={String(sel.length)}
                whenTouched={() => this.props.controller.goTo('selection')}
                tooltip={STRINGS.gotoSelection}
            />
        </Cell>
    }
}

class ClearSelectionCell extends gws.View<FsSearchProps> {
    render() {
        if (this.props.alkisFsSelection.length === 0)
            return null;

        return <Cell>
            <gws.ui.IconButton
                className="modAlkisClearSelectionButton"
                whenTouched={() => this.props.controller.clearSelection()}
                tooltip={STRINGS.clearSelection}
            />
        </Cell>
    }
}

class BackCell extends gws.View<FsSearchProps> {
    render() {
        return <Cell>
            <gws.ui.BackButton
                whenTouched={() => this.props.controller.goBack()}
                tooltip={STRINGS.goBack}

            />
        </Cell>
    }
}

class Bar2 extends React.PureComponent<{}> {
    render() {
        return <Row className="modSidebarSecondaryToolbar">{this.props.children}</Row>
    }
}

class LoaderTab extends React.PureComponent<{}> {
    render() {
        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={STRINGS.formTitle}/>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                <div className="modAlkisEmptyTab">
                    {STRINGS.loading}
                    <gws.ui.Loader/>
                </div>
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

class ErrorTab extends gws.View<FsSearchProps> {
    render() {
        let cc = this.props.controller;

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
                <Bar2>
                    <BackCell {...this.props}/>
                    <Cell flex/>
                    <GotoSelectionCell {...this.props}/>
                </Bar2>
            </sidebar.TabFooter>

        </sidebar.Tab>
    }
}

class SearchForm extends gws.View<FsSearchProps> {

    render() {
        let cc = this.props.controller,
            setup = this.props.alkisFsSetup;

        let boundTo = param => ({
            value: this.props.alkisFsParams[param],
            whenChanged: value => cc.update({
                alkisFsParams: {
                    ...this.props.alkisFsParams,
                    [param]: value
                }
            }),
            whenEntered: submit
        });

        let submit = () => cc.search();

        let clear = () => cc.update({
            alkisFsParams: {},
            lensGeometryType: null,
            marker: null,
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
                        placeholder={STRINGS.vnum}
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
                <Cell width={80}>
                    <gws.ui.TextButton
                        primary
                        whenTouched={submit}
                    >{STRINGS.submitButton}</gws.ui.TextButton>
                </Cell>
                <Cell width={80}>
                    <gws.ui.TextButton
                        whenTouched={clear}
                    >{STRINGS.resetButton}</gws.ui.TextButton>
                </Cell>
            </Row>
        </Form>;
    }
}

class SearchToolbar extends gws.View<FsSearchProps> {
    render() {

        let submit = geom => {
            this.props.alkisFsParams.shape = this.props.controller.map.geom2shape(geom);
            this.props.controller.search();
        };

        let clicked = geometryType => {
            this.props.controller.update({
                lensGeometryType: geometryType,
                lensCallback: geom => submit(geom)
            })
        };

        let cancel = () => {
            this.props.controller.update({
                lensGeometryType: null,
                lensCallback: null
            })
            this.props.alkisFsParams.shape = null;
        };

        let button = gt => <gws.ui.IconButton
            {...gws.tools.cls('modLensButton' + gt, gt === this.props.lensGeometryType && 'isActive')}
            whenTouched={() => clicked(gt)}
        />;

        return <div className="modAlkisLensToolbar">
            <Row>
                <Cell flex/>
                {button('Point')}
                {button('LineString')}
                {button('Polygon')}
                {button('Circle')}
                <Cell>
                    <gws.ui.IconButton
                        className="modLensCancelButton"
                        whenTouched={() => cancel()}
                    />
                </Cell>


            </Row>
        </div>;
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
                <SearchToolbar {...this.props} />
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <Bar2>
                    <Cell flex/>
                    <GotoSelectionCell {...this.props}/>
                </Bar2>
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
        let cc = this.props.controller,
            features = this.props.alkisFsResults,
            empty = features.length === 0;

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
                {
                    empty
                        ? <div className='modAlkisEmptyTab'>
                            {STRINGS.notFound}
                            <a onClick={() => cc.goTo('form')}>{STRINGS.backToForm}</a>
                        </div>

                        : <gws.components.feature.List
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
                }
            </sidebar.TabBody>
            <sidebar.TabFooter>
                <Bar2>
                    <BackCell {...this.props} />
                    <Cell flex/>
                    <HighlightCell {...this.props} features={features}/>
                    <SelectAllCell {...this.props} features={features}/>
                    <ExportCell {...this.props} features={features}/>
                    <PrintCell {...this.props} features={features}/>
                    <GotoSelectionCell {...this.props} />
                </Bar2>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class SelectionTab extends gws.View<FsSearchProps> {
    render() {
        let cc = this.props.controller,
            features = this.props.alkisFsSelection,
            empty = features.length === 0;

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
                {
                    empty

                        ? <div className='modAlkisEmptyTab'>
                            {STRINGS.noSelection}
                        </div>

                        : <gws.components.feature.List
                            controller={cc}
                            features={features}
                            item={f => <gws.ui.Link
                                whenTouched={() => cc.showDetails(f)}
                                content={f.props.teaser}
                            />}

                            leftIcon={leftIcon}
                            rightIcon={rightIcon}
                        />
                }
            </sidebar.TabBody>
            <sidebar.TabFooter>
                <Bar2>
                    <BackCell {...this.props} />
                    <Cell flex/>
                    <HighlightCell {...this.props} features={features}/>
                    <ExportCell {...this.props} features={features}/>
                    <PrintCell {...this.props} features={features}/>
                    <ClearSelectionCell {...this.props} />
                </Bar2>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class DetailsTab extends gws.View<FsSearchProps> {
    render() {
        let cc = this.props.controller;
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
                <Bar2>
                    <BackCell {...this.props} />
                    <Cell flex/>
                    <PrintCell {...this.props} features={[feature]}/>
                    <ToggleSelectionCell {...this.props} features={[feature]}/>
                    <GotoSelectionCell {...this.props} />
                </Bar2>
            </sidebar.TabFooter>
        </sidebar.Tab>

    }
}

class ExportTab extends gws.View<FsSearchProps> {
    render() {
        let groups = this.props.alkisFsExportGroups;

        let changed = (group, value) => this.props.controller.update({
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
                                    whenTouched={() => this.props.controller.submitExport(this.props.alkisFsExportFeatures)}
                                >{STRINGS.exportButton}</gws.ui.TextButton>
                            </Cell>
                        </Row>
                    </Form>
                </div>
            </sidebar.TabBody>
            <sidebar.TabFooter>
                <Bar2>
                    <BackCell {...this.props} />
                    <Cell flex/>
                    <GotoSelectionCell {...this.props} />
                </Bar2>
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

class AlkisController extends gws.Controller implements gws.types.ISidebarItem {

    setup: gws.api.AlkisFsSetupResponse;
    history: Array<string>;

    update(args) {
        super.update(args)
        console.log('UPDATE', args)
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

        let params = {
            ...this.getValue('alkisFsParams'),
            gemarkungUid: value,
            strasse: ''
        };

        this.update({
            alkisFsParams: params,
            alkisFsStrassen: strassen
        });
    };

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

        return this.goTo('list');
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

        let q = {
            ...this.paramsForFeatures(fs),
            printParams: await this.map.printParams(null, this.setup.printTemplate, 0),
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
        this.history.push(this.getValue('alkisFsTab'));
        this.update({
            alkisFsTab: tab
        })
    }

    goBack() {
        console.log(this.history)
        let tab = this.history.pop() || 'form';
        this.update({
            alkisFsTab: tab
        })
    }

    get iconClass() {
        return 'modAlkisSidebarIcon';
    }

    get tooltip() {
        return this.__('modAlkisTooltip');
    }

    get tabView() {
        if (!this.setup)
            return <div className="modSidebarEmptyTab">
                {STRINGS.noData}
            </div>;

        return this.createElement(
            this.connect(SidebarTab, [
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
                    'lensGeometryType',
                ]
            ));
    }
}

export const tags = {
    'Sidebar.Alkis': AlkisController,
};

