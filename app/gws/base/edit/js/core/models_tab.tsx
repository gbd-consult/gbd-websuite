import * as React from 'react';

import * as gws from 'gws';
import * as sidebar from 'gws/elements/sidebar';
import * as components from 'gws/components';
import * as types from './types';
import type {Controller} from './controller';

let {Form, Row, Cell, VBox, VRow} = gws.ui.Layout;


export class ModelsTab extends gws.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    async whenItemTouched(model: gws.types.IModel) {
        this.master().selectModelInSidebar(model)
    }

    async whenRightButtonTouched(model: gws.types.IModel) {
        this.master().selectModelInTableView(model)
    }

    render() {
        let cc = this.master();
        let items = cc.models.filter(m => m.isEditable);

        items.sort((a, b) => a.title.localeCompare(b.title));

        if (gws.lib.isEmpty(items)) {
            return <sidebar.EmptyTab>
                {this.__('editNoLayer')}
            </sidebar.EmptyTab>;
        }

        return <sidebar.Tab className="editSidebar">
            <sidebar.TabHeader>
                <Row>
                    <Cell>
                        <gws.ui.Title content={this.__('editTitle')}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <VBox>
                    <VRow flex>
                        <components.list.List
                            controller={this.props.controller}
                            items={items}
                            content={model => <gws.ui.Link
                                whenTouched={() => this.whenItemTouched(model)}
                                content={model.title}
                            />}
                            uid={model => model.uid}
                            leftButton={model => <components.list.Button
                                className="editModelButton"
                                tooltip={this.__('editOpenModel')}
                                whenTouched={() => this.whenItemTouched(model)}
                            />}
                            rightButton={model => model.hasTableView && <components.list.Button
                                className="editTableViewButton"
                                tooltip={this.__('editTableViewButton')}
                                whenTouched={() => this.whenRightButtonTouched(model)}
                            />}
                        />
                    </VRow>
                </VBox>
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}
