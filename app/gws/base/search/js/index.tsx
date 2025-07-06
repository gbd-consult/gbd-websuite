import * as React from 'react';

import * as gc from 'gc';
import * as sidebar from 'gc/elements/sidebar';
import * as components from 'gc/components';

const MASTER = 'Shared.Search';

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as SearchController;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as SearchController;
}

let { Row, Cell } = gc.ui.Layout;

const SEARCH_DEBOUNCE = 1000;

interface SearchViewProps extends gc.types.ViewProps {
    searchCategories: Array<string>;
    searchInput: string;
    searchOptionsOpen: boolean;
    searchResults: Array<gc.types.IFeature>;
    searchResultsOpen: boolean;
    searchSelectedCategories: Array<string>;
    searchWaiting: boolean;
}

const SearchStoreKeys = [
    'searchCategories',
    'searchInput',
    'searchOptionsOpen',
    'searchResults',
    'searchResultsOpen',
    'searchSelectedCategories',
    'searchWaiting',
];

class SearchResults extends gc.View<SearchViewProps> {
    render() {
        let cc = _master(this);
        if (!this.props.searchResultsOpen) {
            return null;
        }

        if (this.props.searchWaiting) {
            return <div className="searchResults">
                <gc.ui.Loader />
            </div>
        }
        
        let fs = this.props.searchResults || [];

        if (this.props.searchCategories && this.props.searchCategories.length > 0) {
            let cats = this.props.searchSelectedCategories || [];
            fs = fs.filter(f => cats.indexOf(f.category) >= 0);
        }

        if (fs.length === 0) {
            return <div className="searchResults">
                <div className="searchResultsEmpty">
                    <gc.ui.Text content={this.__('searchNoResults')} />
                </div>
            </div>
        }

        fs.sort((a, b) => {
            let ka = a.views.teaser || a.views.title;
            let kb = b.views.teaser || b.views.title;
            return ka.localeCompare(kb)
        });

        let zoomTo = f => this.props.controller.update({
            marker: {
                features: [f],
                mode: 'zoom draw fade'
            }
        });

        let leftButton = f => {
            if (f.geometry)
                return <components.list.Button
                    className="cmpListZoomListButton"
                    whenTouched={() => zoomTo(f)}
                />
            else
                return <components.list.Button
                    className="cmpListDefaultListButton"
                    whenTouched={() => cc.whenFeatureTouched(f)}
                />
        }

        let content = f => {
            if (f.views.teaser)
                return <gc.ui.TextBlock
                    className="searchResultsTeaser"
                    withHTML
                    whenTouched={() => cc.whenFeatureTouched(f)}
                    content={f.views.teaser}
                />
            if (f.views.title)
                return <gc.ui.Link
                    whenTouched={() => cc.whenFeatureTouched(f)}
                    content={f.views.title}
                />
        }

        return <div className="searchResults">
            <components.feature.List
                controller={this.props.controller}
                features={fs}
                content={content}
                leftButton={leftButton}
            />
        </div>;
    }
}

class SearchInputBox extends gc.View<SearchViewProps> {
    sideButton() {
        if (this.props.searchWaiting) {
            return null;
        }

        if (this.props.searchInput) {
            return <gc.ui.Button
                className="searchClearButton"
                tooltip={this.__('searchClearButton')}
                whenTouched={() => _master(this).whenClearButtonTouched()}
            />
        }
    }
    
    render() {
        let cc = _master(this);

        return <Row>
            <Cell>
                <gc.ui.Button className='searchIcon'
                    whenTouched={() => cc.whenSearchIconTouched()}
                />
            </Cell>
            <Cell flex>
                <gc.ui.TextInput
                    value={this.props.searchInput}
                    placeholder={this.__('searchPlaceholder')}
                    whenChanged={val => cc.whenSearchChanged(val)}
                />
            </Cell>
            <Cell className='searchSideButton'>{this.sideButton()}</Cell>
        </Row>

    }
}

class SearchOptions extends gc.View<SearchViewProps> {
    render() {
        let cc = _master(this);
        let selected = this.props.searchSelectedCategories || [];

        return <div className="searchOptions">
            <Row>
                <Cell flex>
                    <gc.ui.Toggle
                        inline
                        type="checkbox"
                        label={this.__('searchCategoryAll')}
                        value={cc.allCategoriesSelected()}
                        whenChanged={v => cc.whenCategoryAllChanged(v)}
                    />
                </Cell>
            </Row>
            {this.props.searchCategories.map((cat, n) =>
                <Row key={n}>
                    <gc.ui.Toggle
                        inline
                        type="checkbox"
                        label={cat}
                        value={selected.indexOf(cat) >= 0}
                        whenChanged={v => cc.whenCategoryChanged(cat, v)}
                    />
                </Row>
            )}
        </div>

    }
}

class SearchBox extends gc.View<SearchViewProps> {

    render() {

        let hasOptions = this.props.searchCategories.length > 1;
        let cls = 'searchBox';

        if (hasOptions) {
            cls += ' hasOptions';
        }
        if (this.props.searchOptionsOpen) {
            cls += ' withOptions';
        }
        if (this.props.searchResultsOpen) {
            cls += ' withResults';
        }

        return <div className={cls}>
            <SearchInputBox {...this.props} />
            {hasOptions && <SearchOptions {...this.props} />}
        </div>;
    }
}

class SearchSidebarView extends gc.View<SearchViewProps> {
    render() {
        return <sidebar.Tab className="searchSidebar">

            <sidebar.TabHeader>
                <SearchBox {...this.props} />
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {this.props.searchResultsOpen && <SearchResults {...this.props} />}
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

class SearchAltbarView extends gc.View<SearchViewProps> {
    render() {
        return <React.Fragment>
            <div className="searchAltbar">
                <SearchBox {...this.props} />
                {this.props.searchResultsOpen && <div className="searchAltbarDropDown">
                    <SearchResults {...this.props} />
                </div>}
            </div>
        </React.Fragment>
    }
}

class SearchAltbar extends gc.Controller {
    get defaultView() {
        return this.createElement(
            this.connect(SearchAltbarView, SearchStoreKeys));
    }
}

class SearchSidebar extends gc.Controller implements gc.types.ISidebarItem {
    iconClass = 'searchSidebarIcon';

    get tooltip() {
        return this.__('searchSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SearchSidebarView, SearchStoreKeys));
    }
}

class SearchController extends gc.Controller {
    uid = MASTER;
    runTimer = null;
    setup: gc.gws.base.search.action.Props;
    categories: Array<string> = [];

    async init() {
        this.setup = this.app.actionProps('search');
        if (!this.setup)
            return;

        this.update({
            searchCategories: this.setup.categories || [],
            searchSelectedCategories: this.setup.categories || [],
        })

        this.app.whenChanged('searchInput', val => this.runDebounced(val));
    }

    protected async runDebounced(keyword) {
        clearTimeout(this.runTimer);

        keyword = (keyword || '').trim();
        if (!keyword) {
            return;
        }

        this.runTimer = setTimeout(() => this.run(keyword), SEARCH_DEBOUNCE);
    }


    protected async run(keyword) {
        this.update({
            searchWaiting: true,
            searchResultsOpen: true,
        });

        let args: gc.types.IFeatureSearchArgs = {
            keyword: keyword
        }

        let cats = this.getValue('searchCategories');
        if (cats && cats.length) {
            args.withCategories = true;
            args.categories = this.getValue('searchSelectedCategories') || [];
        }

        let features = await this.map.searchForFeatures(args);

        this.update({
            searchWaiting: false,
            searchResults: features,
            searchResultsOpen: true,
            marker: null,
        });

        // if (features.length)
        //     this.update({
        //         marker: null,
        //     });

    }

    whenSearchChanged(value) {
        this.update({
            searchInput: value
        });
    }

    whenClearButtonTouched() {
        this.update({
            searchInput: '',
            searchWaiting: false,
            searchResults: null,
            searchResultsOpen: false,
            marker: null,
        });
    }

    whenFeatureTouched(f) {
        this.update({
            marker: {
                features: [f],
                mode: 'zoom draw',
            },
            infoboxContent: <components.feature.InfoList controller={this} features={[f]} />

        });

    }

    whenSearchIconTouched() {
        this.update({
            searchOptionsOpen: !this.getValue('searchOptionsOpen'),
        });
    }

    whenCategoryAllChanged(v) {
        if (this.allCategoriesSelected()) {
            this.update({
                searchSelectedCategories: [],
            });
        } else {
            this.update({
                searchSelectedCategories: this.getValue('searchCategories') || [],
            });
        }

    }

    whenCategoryChanged(cat: string, v: boolean) {
        let selected = this.getValue('searchSelectedCategories') || [];
        if (selected.indexOf(cat) >= 0)
            selected = selected.filter(u => u !== cat);
        else
            selected = selected.concat([cat]);
        this.update({ searchSelectedCategories: selected });


    }

    allCategoriesSelected() {
        let cats = this.getValue('searchCategories') || [];
        let selected = this.getValue('searchSelectedCategories') || [];
        return cats.length === selected.length;
    }

}

gc.registerTags({
    [MASTER]: SearchController,
    'Sidebar.Search': SearchSidebar,
    'Altbar.Search': SearchAltbar,
});

