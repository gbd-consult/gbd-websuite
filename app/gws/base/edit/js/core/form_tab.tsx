import * as React from 'react';

import * as gws from 'gws';
import * as sidebar from 'gws/elements/sidebar';
import * as components from 'gws/components';
import * as types from './types';
import type {Controller} from './controller';

let {Form, Row, Cell, VBox, VRow} = gws.ui.Layout;

export class FormFields extends gws.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    renderContent() {
        let cc = this.master();
        let es = cc.editState;
        let sf = es.sidebarSelectedFeature;
        let values = sf.currentAttributes();

        return <Form>
            <components.Form
                controller={cc}
                feature={sf}
                values={values}
                model={sf.model}
                errors={es.formErrors}
                widgets={cc.formWidgets(sf, values)}
            />
        </Form>

    }

    render() {
        return <VRow flex>
            <Cell flex>
                {this.renderContent()}
            </Cell>
        </VRow>
    }
}

export class FormButtons extends gws.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    buttons() {
        let cc = this.master();
        let es = cc.editState;
        let sf = es.sidebarSelectedFeature;

        let canSave = sf.isNew || sf.isDirty;
        let canDelete = !sf.isNew;

        return [
            <Cell key={1} spaced>
                <gws.ui.Button
                    {...gws.lib.cls('editSaveButton')}
                    disabled={!canSave}
                    tooltip={this.__('editSave')}
                    whenTouched={() => cc.whenFeatureFormSaveButtonTouched(sf)}
                />
            </Cell>,
            <Cell key={2} spaced>
                <gws.ui.Button
                    {...gws.lib.cls('editResetButton')}
                    disabled={!canSave}
                    tooltip={this.__('editReset')}
                    whenTouched={() => cc.whenFeatureFormResetButtonTouched(sf)}
                />
            </Cell>,
            <Cell key={3} spaced>
                <gws.ui.Button
                    {...gws.lib.cls('editCancelButton')}
                    tooltip={this.__('editCancel')}
                    whenTouched={() => cc.whenFeatureFormCancelButtonTouched()}
                />
            </Cell>,
            <Cell key={4} spaced>
                <gws.ui.Button
                    className="editDeleteButton"
                    disabled={!canDelete}
                    tooltip={this.__('editDelete')}
                    whenTouched={() => cc.whenFeatureFormDeleteButtonTouched(sf)}
                />
            </Cell>,
        ]
    }

    renderContent() {
        let cc = this.master();
        let es = cc.editState;
        let sf = es.sidebarSelectedFeature;
        let values = sf.currentAttributes();
        let geomWidget = cc.geometryButtonWidget(sf, values);


        return <Row>
            {geomWidget && <Cell>{geomWidget}</Cell>}
            <Cell flex/>
            {this.buttons()}
        </Row>
    }

    render() {
        return <VRow>
            <Cell flex>
                {this.renderContent()}
            </Cell>
        </VRow>
    }
}

export class FormError extends gws.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    render() {
        let cc = this.master();
        let es = cc.editState;

        if (!es.formErrors) {
            return null;
        }

        return <VRow>
            <Cell flex>
                <gws.ui.Error text={this.__('editValidationErrorText')}/>
            </Cell>
        </VRow>

    }
}

export class FormHeader extends gws.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    render() {
        let cc = this.master();
        let es = cc.editState;
        let sf = es.sidebarSelectedFeature;

        return <Row>
            <Cell flex>
                <gws.ui.Title content={sf.views.title}/>
            </Cell>
        </Row>
    }
}


export class FormTab extends gws.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    async componentDidMount() {
        let cc = this.master();
        let es = cc.editState;
        let sf = es.sidebarSelectedFeature;

        cc.updateEditState({formErrors: null});

        for (let fld of sf.model.fields) {
            await cc.initWidget(fld);
        }
    }

    render() {
        return <sidebar.Tab className="editSidebar editSidebarFormTab">
            <sidebar.TabHeader>
                <FormHeader {...this.props}/>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                <VBox>
                    <FormFields {...this.props}/>
                    <FormError {...this.props}/>
                    <FormButtons {...this.props}/>
                </VBox>
            </sidebar.TabBody>
        </sidebar.Tab>
    }

}
