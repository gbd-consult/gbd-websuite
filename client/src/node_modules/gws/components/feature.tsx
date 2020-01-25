import * as React from 'react';

import * as types from '../types';
import * as ui from '../ui';
import * as tools from '../tools';
import * as list from './list';

let {Row, Cell} = ui.Layout;

interface BaseListProps extends types.ViewProps {
    controller: types.IController;
    features: Array<types.IMapFeature>;
    content?: (f: types.IMapFeature) => React.ReactNode;
    isSelected?: (f: types.IMapFeature) => boolean;
}

interface ListProps extends BaseListProps {
    leftButton?: (f: types.IMapFeature) => React.ReactNode;
    rightButton?: (f: types.IMapFeature) => React.ReactNode;
    withZoom?: boolean;
}

interface InfoListProps extends BaseListProps {
}

interface InfoListState {
    selectedIndex?: number;
}

interface TaskButtonProps extends types.ViewProps {
    feature: types.IMapFeature;
    source?: string;
}

class FeatureList extends list.List<types.IMapFeature> {
}

export class List extends React.PureComponent<ListProps> {
    render() {
        let zoom = f => this.props.controller.update({
            marker: {
                features: [f],
                mode: 'zoom draw fade'
            }
        });

        return <FeatureList
            controller={this.props.controller}
            items={this.props.features}
            content={(f: types.IMapFeature) => this.props.content ? this.props.content(f) : <ui.Text content={f.elements.title}/>}
            uid={f => f.uid}
            isSelected={this.props.isSelected}
            leftButton={this.props.withZoom
                ? f => <list.Button className="cmpListZoomListButton" whenTouched={() => zoom(f)}/>
                : this.props.leftButton}
            rightButton={this.props.rightButton}
        />;
    }
}

export class TaskButton extends React.PureComponent<TaskButtonProps> {
    render() {
        let cc = this.props.controller;
        let element: HTMLElement;

        let touched = () => cc.update({
            taskContext: {
                feature: this.props.feature,
                source: this.props.source,
                element
            }
        });

        return <ui.Button
            className='cmpFeatureTaskButton'
            tooltip={cc.__('cmpFeatureTaskButton')}
            elementRef={e => element = e}
            whenTouched={touched}
        />

    }
}

export class InfoList extends React.Component<InfoListProps, InfoListState> {

    state = {selectedIndex: 0};

    componentDidMount() {
        this.setState({selectedIndex: 0})
    }

    componentDidUpdate(prevProps: InfoListProps) {
        if (prevProps.features !== this.props.features) {
            console.log('NEW FEATURES');
            this.setState({selectedIndex: 0})
        }
    }

    show(n) {
        this.setState({
            selectedIndex: n
        });
        this.props.controller.update({
            marker: {
                features: [this.props.features[n]],
                mode: 'draw',
            }
        });
    }

    render() {
        let cc = this.props.controller;
        let sel = this.state.selectedIndex || 0;
        let f = this.props.features[sel];

        if (!f)
            return null;

        let len = this.props.features.length;

        let dec = () => this.show(sel === 0 ? len - 1 : sel - 1);
        let inc = () => this.show(sel === len - 1 ? 0 : sel + 1);

        let zoom = f => cc.update({
            marker: {
                features: [f],
                mode: 'zoom draw',
            }
        });

        let close = () => cc.update({
            marker: null,
            infoboxContent: null
        });

        let item = this.props.content;
        if (!item)
            item = f => <ui.TextBlock className="cmpDescription" withHTML content={f.elements.description}/>;

        return <div className="cmpInfoboxContent">
            <div className="cmpInfoboxBody">
                {item(f)}
            </div>
            <div className="cmpInfoboxFooter">
                <Row>
                    {len > 1 && <Cell>
                        <div className="cmpInfoboxPagerText">
                            {this.state.selectedIndex + 1} / {len}
                        </div>
                    </Cell>}
                    <Cell flex/>
                    {len > 1 && <Cell>
                        <ui.Button
                            className='cmpInfoboxPagerBack'
                            tooltip={cc.__('cmpInfoboxPagerBack')}
                            whenTouched={dec}/>
                    </Cell>}
                    {len > 1 && <Cell>
                        <ui.Button
                            className='cmpInfoboxPagerForward'
                            tooltip={cc.__('cmpInfoboxPagerForward')}
                            whenTouched={inc}/>
                    </Cell>}
                    <Cell>
                        <TaskButton controller={this.props.controller} feature={f} source="infobox" />
                    </Cell>

                    <Cell>
                        <ui.Button
                            className='cmpInfoboxCloseButton'
                            tooltip={cc.__('cmpInfoboxCloseButton')}
                            whenTouched={close}
                        />
                    </Cell>
                </Row>
            </div>
        </div>
    }
}
