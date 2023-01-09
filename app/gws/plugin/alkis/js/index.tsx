// import * as React from 'react';
// import * as ol from 'openlayers';
//
// import * as gws from 'gws';
// import * as sidebar from 'gws/elements/sidebar';
// import * as components from 'gws/components';
// import * as lens from 'gws/elements/lens';
// import * as storage from 'gws/elements/storage';
//
// const MASTER = 'Shared.Alkis';
//
// let {Form, Row, Cell} = gws.ui.Layout;
//
// function _master(obj: any) {
//     if (obj.app)
//         return obj.app.controller(MASTER) as Controller;
//     if (obj.props)
//         return obj.props.controller.app.controller(MASTER) as Controller;
// }
//
// const EXPORT_PATH = 'fs_info.csv';
//
// type TabName = 'form' | 'list' | 'details' | 'error' | 'selection' | 'export';
//
// const _PREFIX_GEMARKUNG = '@gemarkung';
// const _PREFIX_GEMEINDE = '@gemeinde';
//
// interface FormValues {
//     bblatt?: string;
//     controlInput?: string;
//     flaecheBis?: string;
//     flaecheVon?: string;
//     gemarkungUid?: string;
//     gemeindeUid?: string;
//     hausnummer?: string;
//     name?: string;
//     strasseUid?: string;
//     vnum?: string;
//     vorname?: string;
//     wantEigentuemer?: boolean;
//     shapes?: Array<gws.api.base.shape.Props>;
// }
//
// interface Strasse {
//     name: string;
//     uid: string;
//     gemarkungUid: string;
//     gemeindeUid: string;
// }
//
// interface Toponyms {
//     gemarkungen: Array<gws.api.plugin.alkis.search.ToponymGemarkung>;
//     gemeinden: Array<gws.api.plugin.alkis.search.ToponymGemeinde>;
//     gemarkungIndex: gws.types.Dict,
//     gemeindeIndex: gws.types.Dict,
//     strassen: Array<Strasse>;
// }
//
// interface ViewProps extends gws.types.ViewProps {
//     controller: Controller;
//
//     alkisTab: TabName;
//
//     alkisFsLoading: boolean;
//     alkisFsError: string;
//
//     alkisFsExportGroupIndexes: Array<number>;
//     alkisFsExportFeatures: Array<gws.types.IFeature>;
//
//     alkisFsFormValues: FormValues;
//
//     alkisFsGemarkungListItems: Array<gws.ui.ListItem>;
//     alkisFsStrasseListItems: Array<gws.ui.ListItem>;
//
//     alkisFsResults: Array<gws.types.IFeature>;
//     alkisFsResultCount: number;
//
//     alkisFsDetailsFeature: gws.types.IFeature;
//     alkisFsDetailsText: string;
//
//     alkisFsSelection: Array<gws.types.IFeature>;
//
//     appActiveTool: string;
//
//     features?: Array<gws.types.IFeature>;
//     showSelection: boolean;
//
// }
//
// interface MessageViewProps extends ViewProps {
//     message?: string;
//     error?: string;
//     withFormLink?: boolean;
// };
//
// const StoreKeys = [
//     'alkisTab',
//     'alkisFsSetup',
//     'alkisFsLoading',
//     'alkisFsError',
//     'alkisFsExportGroupIndexes',
//     'alkisFsExportFeatures',
//     'alkisFsFormValues',
//     'alkisFsStrasseListItems',
//     'alkisFsGemarkungListItems',
//     'alkisFsResults',
//     'alkisFsResultCount',
//     'alkisFsDetailsFeature',
//     'alkisFsDetailsText',
//     'alkisFsSelection',
//     'appActiveTool',
// ];
//
// function featureIn(fs: Array<gws.types.IFeature>, f: gws.types.IFeature) {
//     return fs.some(g => g.uid === f.uid);
// }
//
// class ExportAuxButton extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//
//         if (!cc.setup.ui.useExport || this.props.features.length === 0)
//             return null;
//         return <sidebar.AuxButton
//             className="alkisExportAuxButton"
//             whenTouched={() => cc.startExport(this.props.features)}
//             tooltip={cc.__('alkisExportTitle')}
//         />;
//     }
// }
//
// class PrintAuxButton extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//
//         if (!cc.setup.printTemplate || this.props.features.length === 0)
//             return null;
//         return <sidebar.AuxButton
//             className="alkisPrintAuxButton"
//             whenTouched={() => cc.startPrint(this.props.features)}
//             tooltip={cc.__('alkisPrint')}
//         />;
//     }
// }
//
// class HighlightAuxButton extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//
//         if (!cc.setup.printTemplate || this.props.features.length === 0)
//             return null;
//         return <sidebar.AuxButton
//             className="alkisHighlightAuxButton"
//             whenTouched={() => cc.highlightMany(this.props.features)}
//             tooltip={cc.__('alkisHighlight')}
//         />
//     }
// }
//
// class SelectAuxButton extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//
//         if (!cc.setup.ui.useSelect)
//             return null;
//         return <sidebar.AuxButton
//             className="alkisSelectAuxButton"
//             whenTouched={() => cc.select(this.props.features)}
//             tooltip={cc.__('alkisSelectAll')}
//         />
//     }
// }
//
// class ToggleAuxButton extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//
//         if (!cc.setup.ui.useSelect)
//             return null;
//
//         let feature = this.props.features[0];
//
//         if (cc.isSelected(feature))
//             return <sidebar.AuxButton
//                 className="alkisUnselectAuxButton"
//                 whenTouched={() => cc.unselect([feature])}
//                 tooltip={cc.__('alkisUnselect')}
//             />
//         else
//             return <sidebar.AuxButton
//                 className="alkisSelectAuxButton"
//                 whenTouched={() => cc.select([feature])}
//                 tooltip={cc.__('alkisSelect')}
//             />
//     }
// }
//
// class FormAuxButton extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//
//         return <sidebar.AuxButton
//             {...gws.lib.cls('alkisFormAuxButton', this.props.alkisTab === 'form' && 'isActive')}
//             whenTouched={() => cc.goTo('form')}
//             tooltip={cc.__('alkisGotoForm')}
//         />
//     }
// }
//
// class ListAuxButton extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//
//         return <sidebar.AuxButton
//             {...gws.lib.cls('alkisListAuxButton', this.props.alkisTab === 'list' && 'isActive')}
//             whenTouched={() => cc.goTo('list')}
//             tooltip={cc.__('alkisGotoList')}
//         />
//     }
// }
//
// class SelectionAuxButton extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//
//         if (!cc.setup.ui.useSelect)
//             return null;
//
//         let sel = this.props.alkisFsSelection || [];
//
//         return <sidebar.AuxButton
//             {...gws.lib.cls('alkisSelectionAuxButton', this.props.alkisTab === 'selection' && 'isActive')}
//             badge={sel.length ? String(sel.length) : null}
//             whenTouched={() => cc.goTo('selection')}
//             tooltip={cc.__('alkisGotoSelection')}
//         />
//     }
// }
//
// class ClearAuxButton extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//
//         if (this.props.alkisFsSelection.length === 0)
//             return null;
//
//         return <sidebar.AuxButton
//             className="alkisClearAuxButton"
//             whenTouched={() => cc.clearSelection()}
//             tooltip={cc.__('alkisClearSelection')}
//         />
//     }
// }
//
// class ResetAuxButton extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//
//         return <sidebar.AuxButton
//             className="alkisResetAuxButton"
//             whenTouched={() => cc.reset()}
//             tooltip={cc.__('alkisResetButton')}
//         />
//     }
// }
//
// class Navigation extends gws.View<ViewProps> {
//     render() {
//         return <React.Fragment>
//             <FormAuxButton {...this.props}/>
//             <ListAuxButton {...this.props}/>
//             <SelectionAuxButton {...this.props}/>
//         </React.Fragment>
//     }
// }
//
// class LoaderTab extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//
//         return <sidebar.Tab>
//             <sidebar.TabHeader>
//                 <gws.ui.Title content={cc.__('alkisFormTitle')}/>
//             </sidebar.TabHeader>
//             <sidebar.TabBody>
//                 <div className="alkisLoading">
//                     {cc.__('alkisLoading')}
//                     <gws.ui.Loader/>
//                 </div>
//             </sidebar.TabBody>
//         </sidebar.Tab>
//     }
// }
//
// class MessageTab extends gws.View<MessageViewProps> {
//     render() {
//         let cc = _master(this);
//
//         return <sidebar.Tab>
//             <sidebar.TabHeader>
//                 <gws.ui.Title content={cc.__('alkisFormTitle')}/>
//             </sidebar.TabHeader>
//
//             <sidebar.EmptyTabBody>
//                 {this.props.error && <gws.ui.Error text={this.props.error}/>}
//                 {this.props.message && <gws.ui.Text content={this.props.message}/>}
//                 {this.props.withFormLink && <a onClick={() => cc.goTo('form')}>
//                     {cc.__('alkisBackToForm')}
//                 </a>}
//             </sidebar.EmptyTabBody>
//
//             <sidebar.TabFooter>
//                 <sidebar.AuxToolbar>
//                     <Navigation {...this.props}/>
//                 </sidebar.AuxToolbar>
//             </sidebar.TabFooter>
//
//         </sidebar.Tab>
//     }
// }
//
// class SearchForm extends gws.View<ViewProps> {
//
//     render() {
//         let cc = _master(this),
//             setup = cc.setup,
//             form = this.props.alkisFsFormValues;
//
//         let boundTo = key => ({
//             value: form[key],
//             whenChanged: value => cc.updateForm({[key]: value}),
//             whenEntered: () => cc.search()
//         });
//
//         let nameShowMode = '';
//
//         if (setup.withEigentuemer) {
//             if (!setup.withControl)
//                 nameShowMode = 'enabled';
//             else if (form.wantEigentuemer)
//                 nameShowMode = 'enabled';
//             else
//                 nameShowMode = '';
//         }
//
//         let gemarkungListMode = cc.setup.ui.gemarkungListMode;
//
//         let gemarkungListValue = '';
//
//         if (form.gemarkungUid)
//             gemarkungListValue = _PREFIX_GEMARKUNG + ':' + form.gemarkungUid;
//         else if (form.gemeindeUid)
//             gemarkungListValue = _PREFIX_GEMEINDE + ':' + form.gemeindeUid;
//
//         let strasseSearchMode = {
//             anySubstring: setup.ui.strasseSearchMode === gws.api.plugin.alkis.UiStrasseSearchMode.any,
//             withExtra: true,
//             caseSensitive: false,
//         }
//
//
//         return <Form>
//             {nameShowMode && <Row>
//                 <Cell flex>
//                     <gws.ui.TextInput
//                         placeholder={cc.__('alkisVorname')}
//                         disabled={nameShowMode === 'disabled'}
//                         {...boundTo('vorname')}
//                         withClear
//                     />
//                 </Cell>
//             </Row>}
//
//             {nameShowMode && <Row>
//                 <Cell flex>
//                     <gws.ui.TextInput
//                         placeholder={cc.__('alkisNachname')}
//                         disabled={nameShowMode === 'disabled'}
//                         {...boundTo('name')}
//                         withClear
//                     />
//                 </Cell>
//             </Row>}
//
//
//             {gemarkungListMode !== gws.api.plugin.alkis.UiGemarkungListMode.none && <Row>
//                 <Cell flex>
//                     <gws.ui.Select
//                         placeholder={
//                             gemarkungListMode === gws.api.plugin.alkis.UiGemarkungListMode.tree
//                                 ? cc.__('alkisGemeindeGemarkung')
//                                 : cc.__('alkisGemarkung')
//                         }
//                         items={this.props.alkisFsGemarkungListItems}
//                         value={gemarkungListValue}
//                         whenChanged={value => cc.whenGemarkungChanged(value)}
//                         withSearch
//                         withClear
//                     />
//                 </Cell>
//             </Row>}
//
//             <Row>
//                 <Cell flex>
//                     <gws.ui.Select
//                         placeholder={cc.__('alkisStrasse')}
//                         items={this.props.alkisFsStrasseListItems}
//                         {...boundTo('strasseUid')}
//                         searchMode={strasseSearchMode}
//                         withSearch
//                         withClear
//                     />
//                 </Cell>
//                 <Cell width={90}>
//                     <gws.ui.TextInput
//                         placeholder={cc.__('alkisNr')}
//                         {...boundTo('hausnummer')}
//                         withClear
//                     />
//                 </Cell>
//             </Row>
//
//             <Row>
//                 <Cell flex>
//                     <gws.ui.TextInput
//                         placeholder={
//                             setup.withFlurnummer
//                                 ? cc.__('alkisVnumFlur')
//                                 : cc.__('alkisVnum')
//                         }
//                         {...boundTo('vnum')}
//                         withClear
//                     />
//                 </Cell>
//             </Row>
//
//             <Row>
//                 <Cell flex>
//                     <gws.ui.TextInput
//                         placeholder={cc.__('alkisAreaFrom')}
//                         {...boundTo('flaecheVon')}
//                         withClear
//                     />
//                 </Cell>
//                 <Cell flex>
//                     <gws.ui.TextInput
//                         placeholder={cc.__('alkisAreaTo')}
//                         {...boundTo('flaecheBis')}
//                         withClear
//                     />
//                 </Cell>
//             </Row>
//
//             {setup.withBuchung && <Row>
//                 <Cell flex>
//                     <gws.ui.TextInput
//                         placeholder={cc.__('alkisBblatt')}
//                         {...boundTo('bblatt')}
//                         withClear
//                     />
//                 </Cell>
//             </Row>}
//
//             {setup.withControl && <Row className='alkisControlToggle'>
//                 <Cell flex>
//                     <gws.ui.Toggle
//                         type="checkbox"
//                         {...boundTo('wantEigentuemer')}
//                         label={cc.__('alkisWantEigentuemer')}
//                         inline={true}
//                     />
//                 </Cell>
//             </Row>}
//
//             {setup.withControl && form.wantEigentuemer && <Row>
//                 <Cell flex>
//                     <gws.ui.TextArea
//                         {...boundTo('controlInput')}
//                         placeholder={cc.__('alkisControlInput')}
//                     />
//                 </Cell>
//             </Row>}
//
//             <Row>
//                 <Cell flex/>
//                 <Cell>
//                     <gws.ui.Button
//                         {...gws.lib.cls('alkisSearchSubmitButton')}
//                         tooltip={cc.__('alkisSubmitButton')}
//                         whenTouched={() => cc.formSearch()}
//                     />
//                 </Cell>
//                 {setup.ui.searchSelection && <Cell>
//                     <gws.ui.Button
//                         {...gws.lib.cls('alkisSearchSelectionButton')}
//                         tooltip={cc.__('alkisSelectionSearchButton')}
//                         whenTouched={() => cc.selectionSearch()}
//                     />
//                 </Cell>}
//                 {setup.ui.searchSpatial && <Cell>
//                     <gws.ui.Button
//                         {...gws.lib.cls('alkisSearchLensButton', this.props.appActiveTool === 'Tool.Alkis.Lens' && 'isActive')}
//                         tooltip={cc.__('alkisLensButton')}
//                         whenTouched={() => cc.startLens()}
//                     />
//                 </Cell>}
//                 {setup.ui.usePick && <Cell>
//                     <gws.ui.Button
//                         {...gws.lib.cls('alkisPickButton', this.props.appActiveTool === 'Tool.Alkis.Pick' && 'isActive')}
//                         tooltip={cc.__('alkisPickButton')}
//                         whenTouched={() => cc.startPick()}
//                     />
//                 </Cell>}
//                 <Cell>
//                     <gws.ui.Button
//                         {...gws.lib.cls('alkisSearchResetButton')}
//                         tooltip={cc.__('alkisResetButton')}
//                         whenTouched={() => cc.reset()}
//                     />
//                 </Cell>
//             </Row>
//         </Form>;
//     }
// }
//
// class FormTab extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//
//         return <sidebar.Tab>
//             <sidebar.TabHeader>
//                 <gws.ui.Title content={cc.__('alkisFormTitle')}/>
//             </sidebar.TabHeader>
//
//             <sidebar.TabBody>
//                 <SearchForm {...this.props} />
//             </sidebar.TabBody>
//
//             <sidebar.TabFooter>
//                 <sidebar.AuxToolbar>
//                     <Navigation {...this.props}/>
//                 </sidebar.AuxToolbar>
//             </sidebar.TabFooter>
//
//
//         </sidebar.Tab>
//     }
// }
//
// class FeatureList extends gws.View<ViewProps> {
//
//     render() {
//         let cc = _master(this);
//
//         let rightButton = f => cc.isSelected(f)
//             ? <components.list.Button
//                 className="alkisUnselectListButton"
//                 whenTouched={() => cc.unselect([f])}
//             />
//
//             : <components.list.Button
//                 className="alkisSelectListButton"
//                 whenTouched={() => cc.select([f])}
//             />
//         ;
//
//         if (!cc.setup.ui.useSelect)
//             rightButton = null;
//
//         let content = f => <gws.ui.Link
//             whenTouched={() => cc.showDetails(f)}
//             content={f.elements.teaser}
//         />;
//
//         return <components.feature.List
//             controller={cc}
//             features={this.props.features}
//             content={content}
//             isSelected={f => this.props.showSelection && cc.isSelected(f)}
//             rightButton={rightButton}
//             withZoom
//         />
//
//     }
//
// }
//
// class ListTab extends gws.View<ViewProps> {
//     title() {
//         let cc = _master(this);
//
//         let total = this.props.alkisFsResultCount,
//             disp = this.props.alkisFsResults.length;
//
//         if (!disp)
//             return cc.__('alkisFormTitle');
//
//         let s = (total > disp)
//             ? cc.__('alkisSearchResultsPartial')
//             : cc.__('alkisSearchResults');
//
//         s = s.replace(/\$1/g, String(disp));
//         s = s.replace(/\$2/g, String(total));
//
//         return s;
//     }
//
//     render() {
//         let cc = _master(this);
//         let features = this.props.alkisFsResults;
//
//         if (gws.lib.isEmpty(features)) {
//             return <MessageTab
//                 {...this.props}
//                 message={cc.__('alkisNotFound')}
//                 withFormLink={true}
//             />;
//         }
//
//         return <sidebar.Tab>
//             <sidebar.TabHeader>
//                 <gws.ui.Title content={this.title()}/>
//             </sidebar.TabHeader>
//
//             <sidebar.TabBody>
//                 <FeatureList {...this.props} features={features} showSelection={true}/>
//             </sidebar.TabBody>
//
//             <sidebar.TabFooter>
//                 <sidebar.AuxToolbar>
//                     <Navigation {...this.props}/>
//                     <Cell flex/>
//                     <SelectAuxButton {...this.props} features={features}/>
//                     <ExportAuxButton {...this.props} features={features}/>
//                     <PrintAuxButton {...this.props} features={features}/>
//                     <ResetAuxButton {...this.props} features={features}/>
//                 </sidebar.AuxToolbar>
//             </sidebar.TabFooter>
//         </sidebar.Tab>
//     }
// }
//
// class SelectionTab extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//         let features = this.props.alkisFsSelection;
//         let hasFeatures = !gws.lib.isEmpty(features);
//
//         return <sidebar.Tab>
//             <sidebar.TabHeader>
//                 <gws.ui.Title content={cc.__('alkisSelectionTitle')}/>
//             </sidebar.TabHeader>
//
//             <sidebar.TabBody>
//                 {hasFeatures
//                     ? <FeatureList {...this.props} features={features} showSelection={false}/>
//                     : <sidebar.EmptyTabBody>{cc.__('alkisNoSelection')}</sidebar.EmptyTabBody>
//                 }
//             </sidebar.TabBody>
//
//             <sidebar.TabFooter>
//                 <sidebar.AuxToolbar>
//                     <Navigation {...this.props}/>
//                     <Cell flex/>
//                     {hasFeatures && <ExportAuxButton {...this.props} features={features}/>}
//                     {hasFeatures && <PrintAuxButton {...this.props} features={features}/>}
//                     <storage.AuxButtons
//                         controller={cc}
//                         actionName="alkissearchStorage"
//                         hasData={hasFeatures}
//                         dataWriter={name => cc.storageWriter()}
//                         dataReader={(name, data) => cc.storageReader(data)}
//                     />
//                     {hasFeatures && <ClearAuxButton {...this.props} />}
//                 </sidebar.AuxToolbar>
//             </sidebar.TabFooter>
//         </sidebar.Tab>
//     }
// }
//
// class DetailsTab extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//         let feature = this.props.alkisFsDetailsFeature;
//
//         return <sidebar.Tab>
//
//             <sidebar.TabHeader>
//                 <gws.ui.Title content={cc.__('alkisInfoTitle')}/>
//             </sidebar.TabHeader>
//
//             <sidebar.TabBody>
//                 <components.Description content={feature.elements.description}/>
//             </sidebar.TabBody>
//
//             <sidebar.TabFooter>
//                 <sidebar.AuxToolbar>
//                     <Navigation {...this.props}/>
//                     <Cell flex/>
//                     <ToggleAuxButton {...this.props} features={[feature]}/>
//                     <PrintAuxButton {...this.props} features={[feature]}/>
//                     <Cell>
//                         <components.feature.TaskButton
//                             controller={this.props.controller}
//                             feature={feature}
//                             source="alkis"
//                         />
//                     </Cell>
//                 </sidebar.AuxToolbar>
//             </sidebar.TabFooter>
//
//         </sidebar.Tab>
//
//     }
// }
//
// class ExportTab extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//
//         let allGroups = cc.setup.exportGroups,
//             selectedGroupIndexes = this.props.alkisFsExportGroupIndexes;
//
//         let changed = (gid, value) => cc.update({
//             alkisFsExportGroupIndexes: selectedGroupIndexes.filter(g => g !== gid).concat(value ? [gid] : [])
//         });
//
//         return <sidebar.Tab>
//             <sidebar.TabHeader>
//                 <gws.ui.Title content={cc.__('alkisExportTitle')}/>
//             </sidebar.TabHeader>
//             <sidebar.TabBody>
//                 <div className="alkisFsDetailsTabContent">
//                     <Form>
//                         <Row>
//                             <Cell flex>
//                                 <gws.ui.Group vertical>
//                                     {allGroups.map(g => <gws.ui.Toggle
//                                             key={g.index}
//                                             type="checkbox"
//                                             inline
//                                             label={g.title}
//                                             value={selectedGroupIndexes.includes(g.index)}
//                                             whenChanged={value => changed(g.index, value)}
//                                         />
//                                     )}
//                                 </gws.ui.Group>
//                             </Cell>
//                         </Row>
//                         <Row>
//                             <Cell flex/>
//                             <Cell width={120}>
//                                 <gws.ui.Button
//                                     primary
//                                     whenTouched={() => cc.submitExport()}
//                                     label={cc.__('alkisExportButton')}
//                                 />
//                             </Cell>
//                         </Row>
//                     </Form>
//                 </div>
//             </sidebar.TabBody>
//             <sidebar.TabFooter>
//                 <sidebar.AuxToolbar>
//                     <Navigation {...this.props}/>
//                 </sidebar.AuxToolbar>
//             </sidebar.TabFooter>
//         </sidebar.Tab>
//
//     }
// }
//
// class SidebarView extends gws.View<ViewProps> {
//     render() {
//         let cc = _master(this);
//
//         if (!cc.setup)
//             return <sidebar.EmptyTab>{cc.__('alkisNoData')}</sidebar.EmptyTab>;
//
//         if (this.props.alkisFsLoading)
//             return <LoaderTab {...this.props}/>;
//
//         let tab = this.props.alkisTab;
//
//         if (!tab || tab === 'form')
//             return <FormTab {...this.props} />;
//
//         if (tab === 'list')
//             return <ListTab {...this.props} />;
//
//         if (tab === 'details')
//             return <DetailsTab {...this.props} />;
//
//         if (tab === 'error')
//             return <MessageTab {...this.props} error={this.props.alkisFsError} withFormLink={true}/>;
//
//         if (tab === 'selection')
//             return <SelectionTab {...this.props} />;
//
//         if (tab === 'export')
//             return <ExportTab {...this.props} />;
//     }
// }
//
// class Sidebar extends gws.Controller implements gws.types.ISidebarItem {
//     iconClass = 'alkisSidebarIcon';
//
//     get tooltip() {
//         return this.__('alkisSidebarTitle');
//     }
//
//     get tabView() {
//         return this.createElement(
//             this.connect(SidebarView, StoreKeys));
//     }
//
// }
//
// class LensTool extends lens.Tool {
//     title = this.__('alkisLensTitle');
//
//     async runSearch(geom) {
//         let cc = _master(this);
//
//         cc.updateForm({shapes: [this.map.geom2shape(geom)]});
//         await cc.search();
//     }
//
//     stop() {
//         let cc = _master(this);
//
//         super.stop();
//         cc.updateForm({shapes: null});
//     }
// }
//
// class PickTool extends gws.Tool {
//     start() {
//         let cc = _master(this);
//
//         this.map.prependInteractions([
//             this.map.pointerInteraction({
//                 whenTouched: evt => cc.pickTouched(evt.coordinate),
//             }),
//         ]);
//     }
//
//     stop() {
//
//     }
// }
//
// class Controller extends gws.Controller {
//     uid = MASTER;
//     history: Array<string>;
//     selectionLayer: gws.types.IFeatureLayer;
//     setup: gws.api.plugin.alkis.search.Props;
//     toponyms: Toponyms;
//
//     async init() {
//         this.setup = this.app.actionSetup('alkissearch');
//         if (!this.setup)
//             return;
//
//         this.history = [];
//
//         console.time('ALKIS:toponyms:load');
//
//         this.toponyms = await this.loadToponyms();
//
//         console.timeEnd('ALKIS:toponyms:load');
//
//         this.update({
//             alkisTab: 'form',
//             alkisFsLoading: false,
//
//             alkisFsFormValues: {},
//             alkisFsExportGroupIndexes: [],
//
//             alkisFsGemarkungListItems: this.gemarkungListItems(),
//             alkisFsStrasseListItems: this.strasseListItems(this.toponyms.strassen),
//
//             alkisFsSearchResponse: null,
//             alkisFsDetailsResponse: null,
//             alkisFsSelection: [],
//         });
//     }
//
//     async loadToponyms(): Promise<Toponyms> {
//         let res = await this.app.server.alkissearchGetToponyms({});
//
//         let t: Toponyms = {
//             gemeinden: res.gemeinden,
//             gemarkungen: res.gemarkungen,
//             gemeindeIndex: {},
//             gemarkungIndex: {},
//             strassen: []
//         };
//
//         for (let g of res.gemeinden) {
//             t.gemeindeIndex[g.uid] = g;
//         }
//         for (let g of res.gemarkungen) {
//             t.gemarkungIndex[g.uid] = g;
//         }
//
//         t.strassen = res.strasseNames.map((name, n) => ({
//             name,
//             uid: String(n),
//             gemarkungUid: res.strasseGemarkungUids[n],
//             gemeindeUid: t.gemarkungIndex[res.strasseGemarkungUids[n]].gemeindeUid,
//         }));
//
//         return t;
//     }
//
//     gemarkungListItems(): Array<gws.ui.ListItem> {
//         let items = [];
//
//         switch (this.setup.ui.gemarkungListMode) {
//
//             case gws.api.plugin.alkis.UiGemarkungListMode.plain:
//                 for (let g of this.toponyms.gemarkungen) {
//                     items.push({
//                         text: g.name,
//                         value: _PREFIX_GEMARKUNG + ':' + g.uid,
//                     })
//                 }
//                 break;
//
//             case gws.api.plugin.alkis.UiGemarkungListMode.combined:
//                 for (let g of this.toponyms.gemarkungen) {
//                     items.push({
//                         text: g.name,
//                         extraText: this.toponyms.gemeindeIndex[g.gemeindeUid].name,
//                         value: _PREFIX_GEMARKUNG + ':' + g.uid,
//                     });
//                 }
//                 break;
//
//             case gws.api.plugin.alkis.UiGemarkungListMode.tree:
//                 for (let gd of this.toponyms.gemeinden) {
//                     items.push({
//                         text: gd.name,
//                         value: _PREFIX_GEMEINDE + ':' + gd.uid,
//                         level: 1,
//                     });
//                     for (let gk of this.toponyms.gemarkungen) {
//                         if (gk.gemeindeUid === gd.uid) {
//                             items.push({
//                                 text: gk.name,
//                                 value: _PREFIX_GEMARKUNG + ':' + gk.uid,
//                                 level: 2,
//                             });
//                         }
//                     }
//                 }
//                 break;
//         }
//
//         return items;
//     }
//
//     strasseListItems(strassen: Array<Strasse>): Array<gws.ui.ListItem> {
//         let strasseStats = {}, ls = [];
//
//         for (let s of strassen) {
//             strasseStats[s.name] = (strasseStats[s.name] || 0) + 1;
//         }
//
//         switch (this.setup.ui.strasseListMode) {
//
//             case gws.api.plugin.alkis.UiStrasseListMode.plain:
//                 ls = strassen.map(s => ({
//                     text: s.name,
//                     value: s.uid,
//                     extraText: '',
//                 }));
//                 break;
//
//             case gws.api.plugin.alkis.UiStrasseListMode.withGemarkung:
//                 ls = strassen.map(s => ({
//                     text: s.name,
//                     value: s.uid,
//                     extraText: this.toponyms.gemarkungIndex[s.gemarkungUid].name,
//                 }));
//                 break;
//
//             case gws.api.plugin.alkis.UiStrasseListMode.withGemarkungIfRepeated:
//                 ls = strassen.map(s => ({
//                     text: s.name,
//                     value: s.uid,
//                     extraText: strasseStats[s.name] > 1
//                         ? this.toponyms.gemarkungIndex[s.gemarkungUid].name
//                         : ''
//                 }));
//                 break;
//
//             case gws.api.plugin.alkis.UiStrasseListMode.withGemeinde:
//                 ls = strassen.map(s => ({
//                     text: s.name,
//                     value: s.uid,
//                     extraText: this.toponyms.gemeindeIndex[s.gemeindeUid].name,
//                 }));
//                 break;
//
//             case gws.api.plugin.alkis.UiStrasseListMode.withGemeindeIfRepeated:
//                 ls = strassen.map(s => ({
//                     text: s.name,
//                     value: s.uid,
//                     extraText: strasseStats[s.name] > 1
//                         ? this.toponyms.gemeindeIndex[s.gemeindeUid].name
//                         : ''
//                 }));
//                 break;
//         }
//
//         return ls.sort((a, b) => a.text.localeCompare(b.text) || a.extraText.localeCompare(b.extraText));
//     }
//
//     async whenGemarkungChanged(value) {
//         let strassen = this.toponyms.strassen,
//             form: FormValues = {
//                 gemarkungUid: null,
//                 gemeindeUid: null,
//                 strasseUid: null,
//             };
//
//         if (value) {
//             let p = value.split(':');
//
//             if (p[0] === _PREFIX_GEMEINDE) {
//                 form.gemeindeUid = p[1];
//                 strassen = this.toponyms.strassen.filter(s => s.gemeindeUid === form.gemeindeUid);
//             }
//             if (p[0] === _PREFIX_GEMARKUNG) {
//                 form.gemarkungUid = p[1];
//                 form.gemeindeUid = this.toponyms.gemarkungIndex[p[1]].gemeindeUid;
//                 strassen = this.toponyms.strassen.filter(s => s.gemarkungUid === form.gemarkungUid);
//             }
//         }
//
//         this.updateForm(form);
//
//         this.update({
//             alkisFsStrasseListItems: this.strasseListItems(strassen),
//         });
//     };
//
//     selectionSearch() {
//         let geoms = this.selectionGeometries();
//         if (geoms) {
//             this.updateForm({shapes: geoms.map(g => this.map.geom2shape(g))});
//             this.search();
//         }
//     }
//
//     selectionGeometries() {
//         let sel = this.getValue('selectFeatures') as Array<gws.types.IFeature>;
//
//         if (sel)
//             return sel.map(f => f.geometry);
//
//         let m = this.getValue('marker');
//         if (m && m.features) {
//             let gs = gws.lib.compact(m.features.map((f: gws.types.IFeature) => f.geometry));
//             if (gs.length > 0) {
//                 return gs
//             }
//         }
//     }
//
//     formSearch() {
//         this.stopTools();
//         this.search();
//         if (this.setup.ui.autoSpatialSearch)
//             this.startLens();
//     }
//
//     startLens() {
//         this.app.startTool('Tool.Alkis.Lens');
//     }
//
//     stopTools() {
//         this.app.stopTool('Tool.Alkis.*');
//     }
//
//     startPick() {
//         this.app.startTool('Tool.Alkis.Pick');
//     }
//
//     async pickTouched(coord: ol.Coordinate) {
//         let pt = new ol.geom.Point(coord);
//
//         let res = await this.app.server.alkissearchFindFlurstueck({
//             shapes: [this.map.geom2shape(pt)],
//         });
//
//         if (res.error) {
//             return;
//         }
//
//         let features = this.map.readFeatures(res.features);
//
//         this.select(features);
//         this.goTo('selection');
//     }
//
//     async search() {
//
//         this.update({alkisFsLoading: true});
//
//         let params = {...this.getValue('alkisFsFormValues')};
//
//         if (!gws.lib.isEmpty(params.strasseUid)) {
//             let strasse = this.toponyms.strassen[Number(params.strasseUid)];
//             if (strasse) {
//                 params.gemarkungUid = strasse.gemarkungUid;
//                 params.strasse = strasse.name;
//             }
//         }
//         delete params.strasseUid;
//
//         let res = await this.app.server.alkissearchFindFlurstueck(params);
//
//         if (res.error) {
//             let msg = this.__('alkisErrorGeneric');
//
//             if (res.status === 400) {
//                 msg = this.__('alkisErrorControl');
//             }
//
//             if (res.status === 409) {
//                 msg = this.__('alkisErrorTooMany').replace(/\$1/g, this.setup.limit);
//             }
//
//             this.update({
//                 alkisFsError: msg,
//             });
//
//             this.update({alkisFsLoading: false});
//             return this.goTo('error');
//         }
//
//         let features = this.map.readFeatures(res.features);
//
//         this.update({
//             alkisFsResults: features,
//             alkisFsResultCount: res.total,
//             marker: {
//                 features,
//                 mode: 'zoom draw',
//             },
//             infoboxContent: null
//         });
//
//         if (features.length === 1) {
//             await this.showDetails(features[0], true);
//         } else {
//             this.goTo('list');
//         }
//
//         this.update({alkisFsLoading: false});
//     }
//
//     paramsForFeatures(fs: Array<gws.types.IFeature>) {
//         let form = this.getValue('alkisFsFormValues');
//         return {
//             wantEigentuemer: form.wantEigentuemer,
//             controlInput: form.controlInput,
//             fsUids: fs.map(f => f.uid),
//         }
//     }
//
//     async showDetails(f: gws.types.IFeature, highlight = true) {
//         let q = this.paramsForFeatures([f]);
//         let res = await this.app.server.alkissearchGetDetails(q);
//         let feature = this.map.readFeature(res.feature);
//
//         if (f) {
//             if (highlight)
//                 this.highlight(f);
//
//             this.update({
//                 alkisFsDetailsFeature: feature,
//             });
//
//             this.goTo('details');
//         }
//     }
//
//     async startPrint(fs: Array<gws.types.IFeature>) {
//         this.update({
//             printJob: {state: gws.api.lib.job.State.init},
//             marker: null,
//         });
//
//         let qualityLevel = 0;
//         let level = this.setup.printTemplate.qualityLevels[qualityLevel];
//         let dpi = level ? level.dpi : 0;
//
//         let mapParams = await this.map.printParams(null, dpi);
//         let printParams: gws.api.base.printer.ParamsWithTemplate = {
//             type: 'template',
//             templateUid: this.setup.printTemplate.uid,
//             qualityLevel,
//             maps: [mapParams]
//         };
//
//         let q = {
//             findParams: this.paramsForFeatures(fs),
//             printParams,
//             highlightStyle: this.app.style.get('.modMarkerFeature').props,
//         };
//
//         this.update({
//             printerJob: await this.app.server.alkissearchPrint(q),
//             printerSnapshotMode: false,
//         });
//     }
//
//     highlightMany(fs: Array<gws.types.IFeature>) {
//         this.update({
//             marker: {
//                 features: fs,
//                 mode: 'zoom draw'
//             }
//         })
//     }
//
//     highlight(f: gws.types.IFeature) {
//         this.update({
//             marker: {
//                 features: [f],
//                 mode: 'zoom draw'
//             }
//         })
//     }
//
//     isSelected(f: gws.types.IFeature) {
//         let sel = this.getValue('alkisFsSelection') || [];
//         return sel.length && featureIn(sel, f);
//
//     }
//
//     select(fs: Array<gws.types.IFeature>) {
//         if (!this.selectionLayer) {
//             this.selectionLayer = this.map.addServiceLayer(new gws.map.layer.FeatureLayer(this.map, {
//                 uid: '_select',
//                 style: '.alkisSelectFeature',
//             }));
//         }
//
//         let sel = this.getValue('alkisFsSelection') || [],
//             add = [];
//
//         fs.forEach(f => {
//             if (!featureIn(sel, f))
//                 add.push(f)
//         });
//
//         this.update({
//             alkisFsSelection: sel.concat(add)
//         });
//
//         this.selectionLayer.clear();
//         this.selectionLayer.addFeatures(this.getValue('alkisFsSelection'));
//     }
//
//     unselect(fs: Array<gws.types.IFeature>) {
//         let sel = this.getValue('alkisFsSelection') || [];
//
//         this.update({
//             alkisFsSelection: sel.filter(f => !featureIn(fs, f))
//         });
//
//         this.selectionLayer.clear();
//         this.selectionLayer.addFeatures(this.getValue('alkisFsSelection'));
//     }
//
//     storageWriter() {
//         let fs = this.getValue('alkisFsSelection');
//         return {
//             selection: fs.map(f => f.uid)
//         }
//     }
//
//     async storageReader(data) {
//         this.clearSelection();
//
//         if (!data || !data.selection)
//             return;
//
//         let res = await this.app.server.alkissearchFindFlurstueck({
//             wantEigentuemer: false,
//             fsUids: data.selection,
//         });
//
//         if (res.error) {
//             return;
//         }
//
//         let features = this.map.readFeatures(res.features);
//
//         this.select(features);
//
//         this.update({
//             marker: {
//                 features,
//                 mode: 'zoom fade',
//             }
//         });
//
//     }
//
//     clearSelection() {
//         this.update({
//             alkisFsSelection: []
//         });
//         this.map.removeLayer(this.selectionLayer);
//         this.selectionLayer = null;
//     }
//
//     clearResults() {
//         this.update({
//             alkisFsResultCount: 0,
//             alkisFsResults: [],
//             marker: null,
//         });
//     }
//
//     reset() {
//         this.update({
//             alkisFsFormValues: {},
//             alkisFsStrasseListItems: [],
//         });
//         this.clearResults();
//         this.stopTools();
//         this.goTo('form')
//     }
//
//     async startExport(fs: Array<gws.types.IFeature>) {
//         this.update({
//             alkisFsExportFeatures: fs
//         });
//         this.goTo('export')
//     }
//
//     async submitExport() {
//         let fs: Array<gws.types.IFeature> = this.getValue('alkisFsExportFeatures');
//
//         let q = {
//             findParams: this.paramsForFeatures(fs),
//             groupIndexes: this.getValue('alkisFsExportGroupIndexes'),
//         };
//
//         // NB: must use binary because csv doesn't neccessary come in utf8
//
//         let res = await this.app.server.alkissearchExport(q, {binary: true});
//
//         if (res.error) {
//             return;
//         }
//
//         let a = document.createElement('a');
//         a.href = window.URL.createObjectURL(new Blob([res.content], {type: res.mime}));
//         a.download = EXPORT_PATH;
//         document.body.appendChild(a);
//         a.click();
//         window.URL.revokeObjectURL(a.href);
//         document.body.removeChild(a);
//     }
//
//     goTo(tab) {
//         if (this.history[this.history.length - 1] !== tab)
//             this.history.push(tab);
//         this.update({
//             alkisTab: tab
//         });
//         if (tab === 'form') {
//             this.updateForm({controlInput: null});
//         }
//     }
//
//     updateForm(obj) {
//         this.update({
//             alkisFsFormValues: {
//                 ...this.getValue('alkisFsFormValues'),
//                 ...obj
//             }
//         });
//     }
// }
//
// gws.registerTags({
//     [MASTER]: Controller,
//     'Sidebar.Alkis': Sidebar,
//     'Tool.Alkis.Lens': LensTool,
//     'Tool.Alkis.Pick': PickTool,
// });
