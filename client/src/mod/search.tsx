import * as React from 'react';

import * as gws from 'gws';
import * as sidebar from './sidebar';

const MASTER = 'Shared.Search';

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as SearchController;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as SearchController;
}

let {Row, Cell} = gws.ui.Layout;

const SEARCH_DEBOUNCE = 1000;

interface SearchViewProps extends gws.types.ViewProps {
    searchInput: string;
    searchResults: Array<gws.types.IFeature>;
    searchWaiting: boolean;
    searchFailed: boolean;
    searchCategories: Array<string>;
    searchSelectedCategories: Array<string>;
    searchOptionsEnabled: boolean;
    searchOptionsOpen: boolean;
}

const SearchStoreKeys = [
    'searchInput',
    'searchResults',
    'searchWaiting',
    'searchFailed',
    'searchCategories',
    'searchSelectedCategories',
    'searchOptionsEnabled',
    'searchOptionsOpen',
];

class SearchResults extends gws.View<SearchViewProps> {
    render() {
        if (!this.props.searchResults || !this.props.searchResults.length)
            return null;

        let fs = [];

        let cats = this.props.searchSelectedCategories;

        if (cats && cats.length)
            fs = this.props.searchResults.filter(f => cats.indexOf(f.category) >= 0)
        else
            fs = this.props.searchResults

        fs.sort((a, b) => {
            let ka = a.elements.teaser || a.elements.title;
            let kb = b.elements.teaser || b.elements.title;
            return ka.localeCompare(kb)
        });

        return <div className="modSearchResults">
            <gws.components.feature.List
                controller={this.props.controller}
                features={fs}
                content={f => <gws.ui.TextBlock
                    className="modSearchResultsFeatureText"
                    withHTML
                    content={f.elements.teaser || f.elements.title}
                />}
                leftButton={f => <gws.components.list.Button
                    className="cmpListZoomListButton"
                    whenTouched={() => _master(this).show(f)}
                />}
            />
        </div>;
    }
}

class SearchBox extends gws.View<SearchViewProps> {
    sideButton() {
        if (this.props.searchWaiting)
            return <gws.ui.Button
                className="modSearchWaitButton"
            />;

        if (this.props.searchInput)
            return <gws.ui.Button
                className="modSearchClearButton"
                tooltip={this.__('modSearchClearButton')}
                whenTouched={() => _master(this).clear()}
            />
    }


    render() {
        let cc = _master(this);

        let selected = this.props.searchSelectedCategories || [];

        let clearSelected = () => cc.update({searchSelectedCategories: []});

        let toggleSelected = cat => {
            if (selected.indexOf(cat) >= 0)
                selected = selected.filter(u => u !== cat);
            else
                selected = selected.concat([cat]);
            cc.update({searchSelectedCategories: selected});
        }

        let hasOptions = this.props.searchOptionsEnabled && this.props.searchCategories.length > 1;
        let optCls = '';

        if (hasOptions)
            optCls = 'withOptions';
        if (hasOptions && this.props.searchOptionsOpen)
            optCls = 'withOptions isOpen';

        return <div {...gws.tools.cls('modSearchBox', optCls)}>
            <Row>
                <Cell>
                    <gws.ui.Button
                        className='modSearchIcon'
                        whenTouched={_ => cc.update({
                            searchOptionsOpen: !this.props.searchOptionsOpen
                        })}
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.TextInput
                        value={this.props.searchInput}
                        placeholder={this.__('modSearchPlaceholder')}
                        whenChanged={val => _master(this).changed(val)}
                    />
                </Cell>
                <Cell className='modSearchSideButton'>{this.sideButton()}</Cell>
            </Row>

            {hasOptions && <Row className="modSearchOptions">
                <Cell flex>
                    <div>
                        <gws.ui.Toggle
                            inline
                            type="checkbox"
                            label={this.__('modSearchCategoryAll')}
                            value={selected.length === 0}
                            whenChanged={_ => clearSelected()}
                        />
                    </div>


                    {this.props.searchCategories.map((cat, n) =>
                        <div key={n}>
                            <gws.ui.Toggle
                                inline
                                type="checkbox"
                                label={cat}
                                value={selected.indexOf(cat) >= 0}
                                whenChanged={_ => toggleSelected(cat)}
                            />
                        </div>
                    )}
                </Cell>
            </Row>}
        </div>;
    }
}

class SearchSidebarView extends gws.View<SearchViewProps> {
    render() {
        return <sidebar.Tab className="modSearchSidebar">

            <sidebar.TabHeader>
                <SearchBox {...this.props} />
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <SearchResults {...this.props} />
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

class SearchAltbarView extends gws.View<SearchViewProps> {
    render() {
        return <React.Fragment>
            <div className="modSearchAltbar">
                <SearchBox {...this.props} />
            </div>
            <div className="modSearchAltbarResults">
                <SearchResults {...this.props} />
            </div>
        </React.Fragment>
    }
}

class SearchAltbar extends gws.Controller {
    get defaultView() {
        return this.createElement(
            this.connect(SearchAltbarView, SearchStoreKeys));
    }
}

class SearchSidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'modSearchSidebarIcon';

    get tooltip() {
        return this.__('modSearchSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SearchSidebarView, SearchStoreKeys));
    }
}

class SearchController extends gws.Controller {
    uid = MASTER;
    timer = null;

    async init() {
        this.update({searchCategories: this.app.project.searchCategories})

        this.app.whenChanged('searchInput', val => {
            clearTimeout(this.timer);

            val = val.trim();
            if (!val) {
                this.clear();
                return;
            }

            this.update({searchWaiting: true});
            this.timer = setTimeout(() => this.run(val), SEARCH_DEBOUNCE);

        });
    }

    protected async run(keyword) {
        this.update({
            searchWaiting: true,
            searchFailed: false
        });

        let features = await this.map.searchForFeatures({keyword});

        this.update({
            searchWaiting: false,
            searchFailed: features.length === 0,
            searchResults: features
        });

        if (features.length)
            this.update({
                marker: null,
            });

    }

    changed(value) {
        this.update({
            searchInput: value
        });
    }

    clear() {
        this.update({
            searchInput: '',
            searchWaiting: false,
            searchFailed: false,
            searchResults: null,
            marker: null,
        });
    }

    show(f) {
        this.update({
            marker: {
                features: [f],
                mode: 'zoom draw',
            },
            infoboxContent: <gws.components.feature.InfoList controller={this} features={[f]}/>

        });

    }

}

export const tags = {
    [MASTER]: SearchController,
    'Sidebar.Search': SearchSidebar,
    'Altbar.Search': SearchAltbar,
};

