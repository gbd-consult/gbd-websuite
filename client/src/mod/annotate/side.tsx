import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from '../common/sidebar';

import * as types from './types';
//import * as draw from './draw';

let {Form, Row, Cell} = gws.ui.Layout;

interface AnnotateSidebarProps extends gws.types.ViewProps {
    controller: types.MasterController;
    mapUpdateCount: number;
    annotateSelectedFeature: types.Feature;
    annotateFeatureForm: types.FeatureFormData;
    appActiveTool: string;
};

const AnnotateSidebarPropsKeys = [
    'mapUpdateCount',
    'annotateSelectedFeature',
    'annotateFeatureForm',
    'appActiveTool',
];

class AnnotateFeatureDetailsToolbar extends gws.View<AnnotateSidebarProps> {
    render() {

        let master = this.props.controller.app.controller(types.MASTER) as types.MasterController;
        let f = this.props.annotateSelectedFeature;

        return <Row>
            <Cell flex/>
        </Row>;

    }
}

class FeatureDetailsForm extends gws.View<AnnotateSidebarProps> {

    formAttributes(f: types.Feature, ff: types.FeatureFormData): Array<gws.components.sheet.Attribute> {
        switch (f.shapeType) {
            case 'Point':
                return [
                    {
                        name: 'x',
                        title: this.__('modAnnotateX'),
                        value: ff.x,
                        editable: true
                    },
                    {
                        name: 'y',
                        title: this.__('modAnnotateY'),
                        value: ff.y,
                        editable: true
                    }];

            case 'Box':
                return [
                    {
                        name: 'x',
                        title: this.__('modAnnotateX'),
                        value: ff.x,
                        editable: true
                    },
                    {
                        name: 'y',
                        title: this.__('modAnnotateY'),
                        value: ff.y,
                        editable: true
                    },
                    {
                        name: 'w',
                        title: this.__('modAnnotateWidth'),
                        value: ff.w,
                        editable: true
                    },
                    {
                        name: 'h',
                        title: this.__('modAnnotateHeight'),
                        value: ff.h,
                        editable: true
                    }];

            case 'Circle':
                return [
                    {
                        name: 'x',
                        title: this.__('modAnnotateX'),
                        value: ff.x,
                        editable: true
                    },
                    {
                        name: 'y',
                        title: this.__('modAnnotateY'),
                        value: ff.y,
                        editable: true
                    },
                    {
                        name: 'radius',
                        title: this.__('modAnnotateRaidus'),
                        value: ff.radius,
                        editable: true
                    }];
            default:
                return [];

        }

    }

    render() {
        let master = this.props.controller.app.controller(types.MASTER) as types.MasterController;
        let f = this.props.annotateSelectedFeature;
        let ff = this.props.annotateFeatureForm;

        let changed = (key, val) => master.update({
            annotateFeatureForm: {
                ...ff,
                [key]: val
            }
        });

        let submit = () => {
            f.updateFromForm(this.props.annotateFeatureForm);
        };

        let data = this.formAttributes(f, ff);

        data.push({
            name: 'labelTemplate',
            title: this.__('modAnnotateLabelEdit'),
            value: ff['labelTemplate'],
            type: 'text',
            editable: true
        });

        let form = <Form>
            <Row>
                <Cell flex>
                    <gws.components.sheet.Editor
                        data={data}
                        whenChanged={changed}
                        whenEntered={submit}
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.IconButton
                        className="modAnnotateUpdateButton"
                        tooltip={this.props.controller.__('modAnnotateUpdateButton')}
                        whenTouched={submit}
                    />
                </Cell>
            </Row>
        </Form>;

        return <div className="modAnnotateFeatureDetails">
            {form}
        </div>
    }
}

class FeatureDetails extends gws.View<AnnotateSidebarProps> {
    render() {
        let master = this.props.controller.app.controller(types.MASTER) as types.MasterController,
            layer = master.layer,
            selectedFeature = this.props.annotateSelectedFeature;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modAnnotateTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <FeatureDetailsForm {...this.props} />
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.SecondaryToolbar>
                    <Cell>
                        <gws.ui.IconButton
                            className="modSidebarSecondaryClose"
                            tooltip={this.__('modAnnotateCloseButton')}
                            whenTouched={() => {
                                master.unselectFeature();
                                master.app.stopTool('Tool.Annotate.Edit');
                            }}
                        />
                    </Cell>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.IconButton
                            {...gws.tools.cls('modAnnotateLensButton')}
                            tooltip={this.props.controller.__('modAnnotateLensButton')}
                            whenTouched={() => master.startLens()}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton
                            className="modAnnotateRemoveButton"
                            tooltip={this.__('modAnnotateRemoveButton')}
                            whenTouched={() => master.removeFeature(selectedFeature)}
                        />
                    </Cell>


                </sidebar.SecondaryToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }

}

class FeatureList extends gws.View<AnnotateSidebarProps> {
    render() {
        let master = this.props.controller.app.controller(types.MASTER) as types.MasterController,
            layer = master.layer,
            selectedFeature = this.props.annotateSelectedFeature,
            features = layer ? layer.features : null;

        if (gws.tools.empty(features)) {
            return <sidebar.EmptyTab>
                {this.__('modAnnotateNotFound')}
            </sidebar.EmptyTab>;
        }

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modAnnotateTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.components.feature.List
                    controller={master}
                    features={features}
                    isSelected={f => f === selectedFeature}

                    item={(f: types.Feature) => <gws.ui.Link
                        whenTouched={() => {
                            master.selectFeature(f, true);
                            master.app.startTool('Tool.Annotate.Edit')
                        }}
                        content={f.label}
                    />}

                    leftIcon={(f: types.Feature) => <gws.ui.IconButton
                        className="cmpFeatureZoomIcon"
                        whenTouched={() => master.zoomFeature(f)}
                    />}
                />
            </sidebar.TabBody>

            <sidebar.TabFooter>


                <sidebar.SecondaryToolbar>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.IconButton
                            {...gws.tools.cls('modAnnotateEditButton', this.props.appActiveTool === 'Tool.Annotate.Edit' && 'isActive')}
                            tooltip={this.__('modAnnotateEditButton')}
                            whenTouched={() => master.app.startTool('Tool.Annotate.Edit')}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton
                            {...gws.tools.cls('modAnnotateDrawButton', this.props.appActiveTool === 'Tool.Annotate.Draw' && 'isActive')}
                            tooltip={this.__('modAnnotateDrawButton')}
                            whenTouched={() => master.app.startTool('Tool.Annotate.Draw')}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton
                            {...gws.tools.cls('modAnnotateClearButton')}
                            tooltip={this.props.controller.__('modAnnotateClearButton')}
                            whenTouched={() => master.clear()}
                        />
                    </Cell>


                </sidebar.SecondaryToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }

}

class AnnotateSidebar extends gws.View<AnnotateSidebarProps> {
    render() {
        if (this.props.annotateSelectedFeature) {
            return <FeatureDetails {...this.props}/>;
        }
        return <FeatureList {...this.props}/>;

    }
}

export class AnnotateSidebarController extends gws.Controller implements gws.types.ISidebarItem {

    get iconClass() {
        return 'modAnnotateSidebarIcon'
    }

    get tooltip() {
        return this.__('modAnnotateSidebarTooltip');
    }

    get tabView() {
        return this.createElement(
            this.connect(AnnotateSidebar, AnnotateSidebarPropsKeys)
        );
    }

}

