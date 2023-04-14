import * as React from 'react';

import * as gws from 'gws';
import * as components from 'gws/components';

let {Cell} = gws.ui;


interface Props extends gws.types.ModelWidgetProps {
    whenNewButtonTouched?: () => void;
    whenLinkButtonTouched?: () => void;
    whenEditButtonTouched?: (feature?: gws.types.IFeature) => void;
    whenUnlinkButtonTouched?: (feature?: gws.types.IFeature) => void;
    whenDeleteButtonTouched?: (feature?: gws.types.IFeature) => void;
}


class View extends gws.View<Props> {
    buttons(selectedFeature) {
        let cc = this.props.controller;
        let field = this.props.field;

        return <React.Fragment>
            {this.props.whenNewButtonTouched && <Cell>
                <gws.ui.Button
                    className='cmpFormListNewButton'
                    tooltip={this.__('widgetFeatureListNewObject')}
                    whenTouched={this.props.whenNewButtonTouched}
                />
            </Cell>}
            {this.props.whenLinkButtonTouched && <Cell>
                <gws.ui.Button
                    className='cmpFormListLinkButton'
                    tooltip={this.__('widgetFeatureListLinkObject')}
                    whenTouched={this.props.whenLinkButtonTouched}
                />
            </Cell>}
            {this.props.whenEditButtonTouched && <Cell>
                <gws.ui.Button
                    className='cmpFormListEditButton'
                    tooltip={this.__('widgetFeatureListEditObject')}
                    disabled={!selectedFeature}
                    whenTouched={() => this.props.whenEditButtonTouched(selectedFeature)}
                />
            </Cell>}
            {this.props.whenUnlinkButtonTouched && <Cell>
                <gws.ui.Button
                    className='cmpFormListUnlinkButton'
                    tooltip={this.__('widgetFeatureListUnlinkObject')}
                    disabled={!selectedFeature}
                    whenTouched={() => this.props.whenUnlinkButtonTouched(selectedFeature)}
                />
            </Cell>}
            {this.props.whenDeleteButtonTouched && <Cell>
                <gws.ui.Button
                    className='cmpFormListDeleteButton'
                    tooltip={this.__('widgetFeatureListDeleteObject')}
                    disabled={!selectedFeature}
                    whenTouched={() => this.props.whenDeleteButtonTouched(selectedFeature)}
                />
            </Cell>}
        </React.Fragment>
    }

    render() {
        let field = this.props.field;
        let features = this.props.values[field.name] || [];
        let cc = this.props.controller;
        let uid = this.props.widgetProps.uid;

        let selectedFeatureUid = cc.getValue('editState')['widget' + uid];

        let selectedFeature = null;
        for (let f of features)
            if (f.uid === selectedFeatureUid)
                selectedFeature = f;

        let zoomTo = f => cc.update({
            marker: {
                features: [f],
                mode: 'zoom draw fade'
            }
        });

        let select = f => cc.updateObject('editState', {
            ['widget' + uid]: f.uid,
        })

        let leftButton = f => {
            if (f.geometryName)
                return <components.list.Button
                    className="cmpListZoomListButton"
                    whenTouched={() => zoomTo(f)}
                />
            else
                return <components.list.Button
                    className="cmpListDefaultListButton"
                    whenTouched={() => select(f)}
                />
        }

        return <div className="cmpFormList">
            <components.feature.List
                controller={this.props.controller}
                features={features}
                content={f => <gws.ui.Link
                    content={f.views.title}
                    whenTouched={() => select(f)}
                />}
                isSelected={f => f.uid === selectedFeatureUid}
                leftButton={leftButton}
            />
            <gws.ui.Row className="cmpFormListToolbar">
                {this.buttons(selectedFeature)}
            </gws.ui.Row>
        </div>

    }
}

class Controller extends gws.Controller {
    view(props) {
        return this.createElement(View, props)
    }
}


gws.registerTags({
    'ModelWidget.featureList': Controller,
})
