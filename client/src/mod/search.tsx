import * as React from 'react';

import * as gws from 'gws';
import * as sidebar from './common/sidebar';

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
    searchResults: Array<gws.types.IMapFeature>;
    searchWaiting: boolean;
    searchFailed: boolean;
}

const SearchStoreKeys = [
    'searchInput',
    'searchResults',
    'searchWaiting',
    'searchFailed',
];

class SearchResults extends gws.View<SearchViewProps> {
    render() {
        if (!this.props.searchResults || !this.props.searchResults.length)
            return null;
        return <div className="modSearchResults">
            <gws.components.feature.List
                controller={this.props.controller}
                features={this.props.searchResults}
                content={f => <gws.ui.TextBlock
                    className="modSearchResultsFeatureText"
                    withHTML
                    whenTouched={() => _master(this).show(f)}
                    content={f.props.teaser}
                />}
            />
        </div>;
    }
}

class SearchBox extends gws.View<SearchViewProps> {
    sideButton() {
        if (this.props.searchWaiting)
            return <gws.ui.IconButton
                className="modSearchWaitButton"
            />;

        if (this.props.searchInput)
            return <gws.ui.IconButton
                className="modSearchClearButton"
                tooltip={this.__('modSearchClearButton')}
                whenTouched={() => _master(this).clear()}
            />
    }

    render() {
        return <div className="modSearchBox">
            <Row>
                <Cell>
                    <gws.ui.IconButton className='modSearchIcon'/>
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
                mode: 'pan draw',
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

