import * as React from 'react';

import * as gws from 'gws';
import * as sidebar from './common/sidebar';

const MASTER = 'Shared.Search';

let {Row, Cell} = gws.ui.Layout;

const SEARCH_DEBOUNCE = 1000;

interface SearchViewProps extends gws.types.ViewProps {
    searchInput: string;
    searchResults: Array<gws.types.IMapFeature>;
    searchWaiting: boolean;
    searchFailed: boolean;
    master: SearchController;
}

const SearchStoreKeys = [
    'searchInput',
    'searchResults',
    'searchWaiting',
    'searchFailed',

]

class SearchResults extends gws.View<SearchViewProps> {
    render() {
        if (!this.props.searchResults || !this.props.searchResults.length)
            return null;
        return <div className="modSearchResults">
            <gws.components.feature.List
                controller={this.props.controller}
                features={this.props.searchResults}
                item={f => <gws.ui.TextBlock
                    className="modSearchResultsFeatureText"
                    withHTML
                    whenTouched={() => this.props.master.zoomTo(f)}
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
                whenTouched={() => this.props.master.clear()}
            />
    }

    render() {
        return <div className="modSearchBox">
            <Row>
                <Cell className='modSearchIcon'/>
                <Cell flex>
                    <gws.ui.TextInput
                        value={this.props.searchInput}
                        placeholder={this.label('modSearchPlaceholder')}
                        whenChanged={val => this.props.master.changed(val)}
                    />
                </Cell>
                <Cell className='modSearchSideButton'>{this.sideButton()}</Cell>
            </Row>
        </div>;
    }
}

class SidebarView extends gws.View<SearchViewProps> {
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

class AltbarSearchView extends gws.View<SearchViewProps> {
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


class AltbarSearch extends gws.Controller {
    get defaultView() {
        let master = this.app.controller(MASTER) as SearchController;
        return this.createElement(
            this.connect(AltbarSearchView, SearchStoreKeys),
            {master});
    }
}

class SidebarSearch extends gws.Controller implements gws.types.ISidebarItem {
    get iconClass() {
        return 'modSearchSidebarIcon'
    }

    get tooltip() {
        return this.__('modSearchTooltip');
    }

    get tabView() {
        let master = this.app.controller(MASTER) as SearchController;
        return this.createElement(
            this.connect(SidebarView, SearchStoreKeys),
            {master});
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

    protected async run(val) {
        this.update({searchWaiting: true, searchFailed: false});

        let params = await this.map.searchParams(val, null);
        let res = await this.app.server.searchFindFeatures(params);

        if (res.error) {
            console.log('SEARCH_ERROR', res);
            return [];
        }

        let features = this.map.readFeatures(res.features);

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
        this.update({searchInput: value});
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

    zoomTo(f) {
        this.update({
            marker: {
                features: [f],
                mode: 'pan draw',
            }
        });

    }

}

export const tags = {
    [MASTER]: SearchController,
    'Sidebar.Search': SidebarSearch,
    'Altbar.Search': AltbarSearch,
};

