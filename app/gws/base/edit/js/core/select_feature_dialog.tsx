import * as React from 'react';

import * as gws from 'gws';
import * as types from './types';
import {FeatureList} from './feature_list';
import type {Controller} from './controller';

export class SelectFeatureDialog extends gws.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    async componentDidMount() {
        let cc = this.master();
        let dd = this.props.editDialogData as types.SelectFeatureDialogData;
        await cc.featureCache.updateForModel(dd.model);
    }

    render() {
        let cc = this.master();
        let dd = this.props.editDialogData as types.SelectFeatureDialogData;
        let searchText = cc.getFeatureListSearchText(dd.model.uid)
        let features = cc.featureCache.getForModel(dd.model);

        let cancelButton = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.closeDialog()}
        />;

        return <gws.ui.Dialog
            className="editSelectFeatureDialog"
            title={this.__('editSelectFeatureTitle')}
            whenClosed={() => cc.closeDialog()}
            buttons={[cancelButton]}
        >
            <FeatureList
                controller={cc}
                whenFeatureTouched={dd.whenFeatureTouched}
                whenSearchChanged={val => cc.whenFeatureListSearchChanged(dd.model, val)}
                features={features}
                searchText={searchText}
                withSearch={dd.model.supportsKeywordSearch}
            />
        </gws.ui.Dialog>;
    }
}

