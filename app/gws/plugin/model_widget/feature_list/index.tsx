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

// see app/gws/plugin/model_field/file/__init__.py
export interface FileOutputProps {
    downloadUrl?: string
    extension?: string
    label?: string
    previewUrl?: string
    size?: number
}


class View extends gws.View<Props> {
    buttons(sf) {
        return <React.Fragment>
            {this.props.widgetProps['withNewButton'] && this.props.whenNewButtonTouched && <Cell>
                <gws.ui.Button
                    className='cmpFormListNewButton'
                    tooltip={this.__('widgetFeatureListNewObject')}
                    whenTouched={this.props.whenNewButtonTouched}
                />
            </Cell>}

            {this.props.widgetProps['withLinkButton'] && this.props.whenLinkButtonTouched && <Cell>
                <gws.ui.Button
                    className='cmpFormListLinkButton'
                    tooltip={this.__('widgetFeatureListLinkObject')}
                    whenTouched={this.props.whenLinkButtonTouched}
                />
            </Cell>}

            {this.props.widgetProps['withEditButton'] && this.props.whenEditButtonTouched && <Cell>
                <gws.ui.Button
                    className='cmpFormListEditButton'
                    tooltip={this.__('widgetFeatureListEditObject')}
                    disabled={!sf}
                    whenTouched={() => this.props.whenEditButtonTouched(sf)}
                />
            </Cell>}

            {this.props.widgetProps['withUnlinkButton'] && this.props.whenUnlinkButtonTouched && <Cell>
                <gws.ui.Button
                    className='cmpFormListUnlinkButton'
                    tooltip={this.__('widgetFeatureListUnlinkObject')}
                    disabled={!sf}
                    whenTouched={() => this.props.whenUnlinkButtonTouched(sf)}
                />
            </Cell>}

            {this.props.widgetProps['withDeleteButton'] && this.props.whenDeleteButtonTouched && <Cell>
                <gws.ui.Button
                    className='cmpFormListDeleteButton'
                    tooltip={this.__('widgetFeatureListDeleteObject')}
                    disabled={!sf}
                    whenTouched={() => this.props.whenDeleteButtonTouched(sf)}
                />
            </Cell>}

            {this.props.widgetProps['toFileField'] && <Cell flex/>}

            {this.props.widgetProps['toFileField'] && <Cell>
                <gws.ui.Button
                    className='cmpFormFileDownloadButton'
                    tooltip={this.__('widgetFeatureListFileDownload')}
                    disabled={!sf}
                    whenTouched={() => this.download(sf)}
                />
            </Cell>}
        </React.Fragment>
    }

    selectedFeature() {
        let cc = this.props.controller;
        let uid = this.props.widgetProps.uid;
        let field = this.props.field;
        let features = this.props.values[field.name] || [];
        let selection = cc.getValue('editState')['widget' + uid] || {};

        for (let feature of features) {
            if (feature.model.uid === selection.modelUid && feature.uid === selection.uid) {
                return feature;
            }
        }
    }

    selectFeature(feature: gws.types.IFeature) {
        let cc = this.props.controller;
        let uid = this.props.widgetProps.uid;

        cc.updateObject('editState', {
            ['widget' + uid]: {modelUid: feature.model.uid, uid: feature.uid}
        });
    }

    download(feature: gws.types.IFeature) {
        let fp = feature.getAttribute(this.props.widgetProps['toFileField']) as FileOutputProps;
        let u = fp.downloadUrl || '';
        gws.lib.downloadUrl(u, u.split('/').pop())
    }

    renderAsFeatureList() {
        let cc = this.props.controller;
        let field = this.props.field;
        let features = this.props.values[field.name] || [];
        let sf = this.selectedFeature();

        let zoomTo = f => cc.update({
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
                    whenTouched={() => this.selectFeature(f)}
                />
        }

        return <div className="cmpFormList">
            <components.feature.List
                controller={this.props.controller}
                features={features}
                content={f => <gws.ui.Link
                    content={f.views.title}
                    whenTouched={() => this.selectFeature(f)}
                />}
                isSelected={f => f === sf}
                leftButton={leftButton}
            />
            <gws.ui.Row className="cmpFormListToolbar">
                {this.buttons(sf)}
            </gws.ui.Row>
        </div>

    }

    renderAsFileList() {
        let cc = this.props.controller;
        let field = this.props.field;
        let features = this.props.values[field.name] || [];
        let sf = this.selectedFeature();

        let items: Array<components.file.FileItem> = [];

        for (let f of features) {
            let fp = f.getAttribute(this.props.widgetProps['toFileField']) as FileOutputProps;
            items.push({
                feature: f,
                label: f.views.title,
                extension: fp.extension,
                previewUrl: fp.previewUrl,
            })
        }

        return <div className="cmpFormList">
            <components.file.FileList
                controller={this.props.controller}
                items={items}
                isSelected={item => item.feature === sf}
                whenTouched={item => this.selectFeature(item.feature)}
            />
            <gws.ui.Row className="cmpFormListToolbar">
                {this.buttons(sf)}
            </gws.ui.Row>
        </div>
    }

    render() {
        if (this.props.widgetProps['toFileField'])
            return this.renderAsFileList()
        return this.renderAsFeatureList();
    }


}

class Controller extends gws.Controller {
    view(props) {
        return this.createElement(View, props)
    }
}


gws.registerTags({
    'ModelWidget.featureList': Controller,
    'ModelWidget.fileList': Controller,
})
