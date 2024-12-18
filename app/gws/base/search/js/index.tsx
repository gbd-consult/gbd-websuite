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

let {Row, Cell} = gc.ui.Layout;

const SEARCH_DEBOUNCE = 1000;

interface SearchViewProps extends gc.types.ViewProps {
    searchInput: string;
    searchResults: Array<gc.types.IFeature>;
    searchWaiting: boolean;
    searchFailed: boolean;
}

const SearchStoreKeys = [
    'searchInput',
    'searchResults',
    'searchWaiting',
    'searchFailed',
];

class SearchResults extends gc.View<SearchViewProps> {
    render() {
        let cc = _master(this);

        if (!this.props.searchResults || !this.props.searchResults.length)
            return null;

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
                features={this.props.searchResults}
                content={content}
                leftButton={leftButton}
            />
        </div>;
    }
}

class SearchBox extends gc.View<SearchViewProps> {
    sideButton() {
        if (this.props.searchWaiting)
            return <gc.ui.Button
                className="searchWaitButton"
            />;

        if (this.props.searchInput)
            return <gc.ui.Button
                className="searchClearButton"
                tooltip={this.__('searchClearButton')}
                whenTouched={() => _master(this).whenClearButtonTouched()}
            />
    }

    render() {
        return <div className="searchBox">
            <Row>
                <Cell>
                    <gc.ui.Button className='searchIcon'/>
                </Cell>
                <Cell flex>
                    <gc.ui.TextInput
                        value={this.props.searchInput}
                        placeholder={this.__('searchPlaceholder')}
                        whenChanged={val => _master(this).whenSearchChanged(val)}
                    />
                </Cell>
                <Cell className='searchSideButton'>{this.sideButton()}</Cell>
            </Row>
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
                <SearchResults {...this.props} />
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

class SearchAltbarView extends gc.View<SearchViewProps> {
    render() {
        return <React.Fragment>
            <div className="searchAltbar">
                <SearchBox {...this.props} />
            </div>
            <div className="searchAltbarResults">
                <SearchResults {...this.props} />
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
    timer = null;

    async init() {
        this.app.whenChanged('searchInput', val => {
            clearTimeout(this.timer);

            val = val.trim();
            if (!val) {
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

    whenSearchChanged(value) {
        this.update({
            searchInput: value
        });
    }

    whenClearButtonTouched() {
        this.update({
            searchInput: '',
            searchWaiting: false,
            searchFailed: false,
            searchResults: null,
            marker: null,
        });
    }

    whenFeatureTouched(f) {
        this.update({
            marker: {
                features: [f],
                mode: 'zoom draw',
            },
            infoboxContent: <components.feature.InfoList controller={this} features={[f]}/>

        });

    }

}

gc.registerTags({
    [MASTER]: SearchController,
    'Sidebar.Search': SearchSidebar,
    'Altbar.Search': SearchAltbar,
});

