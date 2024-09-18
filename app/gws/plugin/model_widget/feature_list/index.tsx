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

interface ListProps extends Props {
    selectedFeature?: gws.types.IFeature;
    whenTouched?: (feature: gws.types.IFeature) => void;
    withZoom?: boolean;
}

class FormView extends gws.View<Props> {
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
        let fp = feature.getAttribute(this.props.widgetProps['toFileField']) as gws.types.ServerFileProps;
        let u = fp.downloadUrl || '';
        gws.lib.downloadUrl(u, u.split('/').pop())
    }

    render() {
        let sf = this.selectedFeature();

        let list = this.props.widgetProps['toFileField']
            ? <FileList
                {...this.props}
                selectedFeature={sf}
                whenTouched={f => this.selectFeature(f)}
            />
            : <FeatureList
                {...this.props}
                selectedFeature={sf}
                whenTouched={f => this.selectFeature(f)}
                withZoom={true}
            />
        ;

        if (this.props.disabled) {
            return <gws.ui.Row className="cmpFormListDisabledMessage">
                    {this.__('editDisabledUntilSave')}
                </gws.ui.Row>

        }

        return <div className="cmpFormList">
            {list}
            <gws.ui.Row className="cmpFormListToolbar">
                {this.buttons(sf)}
            </gws.ui.Row>
        </div>
    }
}

class CellView extends gws.View<Props> {
    render() {

        let field = this.props.field;
        let features = this.props.values[field.name] || [];

        if (features.length === 0)
            return null;

        let list = this.props.widgetProps['toFileField']
            ? <FileList
                {...this.props}
            />
            : <FeatureList
                {...this.props}
                withZoom={false}
            />
        ;

        return <div className="cmpFormList">
            {list}
        </div>
    }
}

class FeatureList extends gws.View<ListProps> {
    render() {
        let cc = this.props.controller;
        let field = this.props.field;
        let features = this.props.values[field.name] || [];
        let touched = this.props.whenTouched || (f => null);

        let zoomTo = f => cc.update({
            marker: {
                features: [f],
                mode: 'zoom draw fade'
            }
        });

        let leftButton = f => {
            if (this.props.withZoom && f.geometryName)
                return <components.list.Button
                    className="cmpListZoomListButton"
                    whenTouched={() => zoomTo(f)}
                />
            else
                return <components.list.Button
                    className="cmpListDefaultListButton"
                    whenTouched={() => touched(f)}
                />
        }

        return <components.feature.List
            controller={this.props.controller}
            features={features}
            content={f => <gws.ui.Link
                content={f.views.title}
                whenTouched={() => touched(f)}
            />}
            isSelected={f => f === this.props.selectedFeature}
            leftButton={leftButton}
        />

    }
}

class FileList extends gws.View<ListProps> {
    render() {
        let field = this.props.field;
        let features = this.props.values[field.name] || [];
        let items: Array<components.file.FileItem> = [];
        let touched = this.props.whenTouched || (f => null);

        for (let f of features) {
            let fp = f.getAttribute(this.props.widgetProps['toFileField']) as gws.types.ServerFileProps;
            items.push({
                feature: f,
                label: f.views.title,
                extension: fp.extension,
                previewUrl: fp.previewUrl,
            })
        }

        return <components.file.FileList
            controller={this.props.controller}
            items={items}
            isSelected={item => item.feature === this.props.selectedFeature}
            whenTouched={item => touched(item.feature)}
        />

    }
}

class Controller extends gws.Controller {
    cellView(props) {
        return this.createElement(CellView, props)
    }

    activeCellView(props) {
        // feature lists no editable in a table
        return this.createElement(CellView, props)
    }

    formView(props) {
        return this.createElement(FormView, props)
    }
}


gws.registerTags({
    'ModelWidget.featureList': Controller,
    'ModelWidget.fileList': Controller,
})
