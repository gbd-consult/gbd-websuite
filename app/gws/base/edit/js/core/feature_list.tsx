import * as React from 'react';

import * as gc from 'gc';
import * as components from 'gc/components';
import * as types from './types';

let {Form, Row, Cell, VBox, VRow} = gc.ui.Layout;

export class FeatureList extends gc.View<types.FeatureListProps> {
    render() {
        let zoomTo = f => this.props.controller.update({
            marker: {
                features: [f],
                mode: 'zoom draw fade'
            }
        });

        let leftButton = f => {
            if (f.geometryName)
                return <components.list.Button
                    className="cmpListZoomListButton"
                    whenTouched={() => zoomTo(f)}
                />
            else
                return <components.list.Button
                    className="cmpListDefaultListButton"
                    whenTouched={() => this.props.whenFeatureTouched(f)}
                />
        }

        return <VBox>
            {this.props.withSearch && <VRow>
                <div className="modSearchBox">
                    <Row>
                        <Cell>
                            <gc.ui.Button className='searchIcon'/>
                        </Cell>
                        <Cell flex>
                            <gc.ui.TextInput
                                placeholder={this.__('editSearchPlaceholder')}
                                withClear={true}
                                value={this.props.searchText}
                                whenChanged={val => this.props.whenSearchChanged(val)}
                            />
                        </Cell>
                    </Row>
                </div>
            </VRow>}
            <VRow flex>
                <components.feature.List
                    controller={this.props.controller}
                    features={this.props.features}
                    content={f => <gc.ui.Link
                        content={f.views.title}
                        whenTouched={() => this.props.whenFeatureTouched(f)}
                    />}
                    leftButton={leftButton}
                />
            </VRow>
        </VBox>


    }
}
