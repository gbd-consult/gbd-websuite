import * as React from 'react';

import * as gws from 'gws';
import * as sidebar from 'gws/elements/sidebar';
import * as components from 'gws/components';
import * as types from './types';
import type {Controller} from './controller';

let {Form, Row, Cell, VBox, VRow} = gws.ui.Layout;

export class FormTab extends gws.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    async whenSaveButtonTouched(feature: gws.types.IFeature) {
        let cc = this.master();
        let ok = await cc.saveFeatureInSidebar(feature);
        if (ok) {
            await cc.closeForm();
        }
    }

    whenDeleteButtonTouched(feature: gws.types.IFeature) {
        let cc = this.master();
        cc.showDialog({
            type: 'DeleteFeature',
            feature,
            whenConfirmed: () => this.whenDeleteConfirmed(feature),
        })
    }

    async whenDeleteConfirmed(feature: gws.types.IFeature) {
        let cc = this.master();
        let ok = await cc.deleteFeature(feature);
        if (ok) {
            await cc.closeDialog();
            await cc.closeForm();
        }
    }

    whenResetButtonTouched(feature: gws.types.IFeature) {
        let cc = this.master();
        let es = cc.editState;
        feature.resetEdits();
        cc.updateEditState();
    }

    async whenCancelButtonTouched() {
        let cc = this.master();
        await cc.closeForm();
    }

    whenWidgetChanged(feature: gws.types.IFeature, field: gws.types.IModelField, value: any) {
        let cc = this.master();
        feature.editAttribute(field.name, value);
        cc.updateEditState();
    }

    whenWidgetEntered(feature: gws.types.IFeature, field: gws.types.IModelField, value: any) {
        let cc = this.master();
        feature.editAttribute(field.name, value);
        cc.updateEditState();
        this.whenSaveButtonTouched(feature);
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
        let cc = this.master();
        let es = cc.editState;
        let sf = es.sidebarSelectedFeature;
        let values = sf.currentAttributes();
        let widgets = [];
        let geomWidget = null;

        for (let fld of sf.model.fields) {
            let w = cc.createWidget(
                gws.types.ModelWidgetMode.form,
                fld,
                sf,
                values,
                this.whenWidgetChanged.bind(this),
                this.whenWidgetEntered.bind(this),
            );
            if (w && fld.widgetProps.type === 'geometry' && !fld.widgetProps.isInline && !geomWidget) {
                geomWidget = w;
                w = null;
            }
            widgets.push(w);
        }

        let canSave = sf.isNew || sf.isDirty;
        let canDelete = !sf.isNew;

        return <sidebar.Tab className="editSidebar editSidebarFormTab">
            <sidebar.TabHeader>
                <Row>
                    <Cell flex>
                        <gws.ui.Title content={sf.views.title}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <VBox>
                    <VRow flex>
                        <Cell flex>
                            <Form>
                                <components.Form
                                    controller={this.props.controller}
                                    feature={sf}
                                    values={values}
                                    model={sf.model}
                                    errors={es.formErrors}
                                    widgets={widgets}
                                />
                            </Form>
                        </Cell>
                    </VRow>
                    {es.formErrors && <VRow>
                        <Row>
                            <Cell flex>
                                <gws.ui.Error text={this.__('editValidationErrorText')}/>
                            </Cell>
                        </Row>
                    </VRow>}
                    <VRow>
                        <Row>
                            {geomWidget && <Cell>{geomWidget}</Cell>}
                            <Cell flex/>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.lib.cls('editSaveButton')}
                                    disabled={!canSave}
                                    tooltip={this.__('editSave')}
                                    whenTouched={() => this.whenSaveButtonTouched(sf)}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.lib.cls('editResetButton')}
                                    disabled={!canSave}
                                    tooltip={this.__('editReset')}
                                    whenTouched={() => this.whenResetButtonTouched(sf)}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.lib.cls('editCancelButton')}
                                    tooltip={this.__('editCancel')}
                                    whenTouched={() => this.whenCancelButtonTouched()}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    className="editDeleteButton"
                                    disabled={!canDelete}
                                    tooltip={this.__('editDelete')}
                                    whenTouched={() => this.whenDeleteButtonTouched(sf)}
                                />
                            </Cell>
                        </Row>
                    </VRow>
                </VBox>
            </sidebar.TabBody>

        </sidebar.Tab>
    }

}
