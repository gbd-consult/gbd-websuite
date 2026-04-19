import * as React from 'react';


import * as gc from 'gc';

import * as components from 'gc/components';
import * as toolbar from 'gc/elements/toolbar';
import { FormField } from 'gc/components/form';

let { VBox, VRow } = gc.ui.Layout;

const MASTER = 'Shared.Exporter';

function _master(obj: any): Controller {
    if (obj.app)
        return obj.app.controller(MASTER) as Controller;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as Controller;
}

const JOB_POLL_INTERVAL = 2000;

interface ViewProps extends gc.types.ViewProps {
    controller: Controller;
    exportJob?: gc.gws.JobStatusResponse;
    exportState?: 'running' | 'error' | 'complete';
}

const StoreKeys = [
    'exportJob',
    'exportState',
];



class Dialog extends gc.View<ViewProps> {

    render() {
        let cc = _master(this);
        let ps = this.props.exportState;
        let job = this.props.exportJob;

        let cancel = () => cc.cancelExport();
        let stop = () => cc.stop();

        if (ps === 'running') {

            let label = '';

            if (job.stepName)
                label = gc.lib.shorten(job.stepName, 40);

            return <gc.ui.Dialog
                className="exporterProgressDialog"
                title={this.__('exporterRunning')}
                whenClosed={cancel}
                buttons={[
                    <gc.ui.Button label={this.__('exporterCancel')} whenTouched={cancel} />
                ]}
            >
                <gc.ui.Progress value={job.progress} />
                <gc.ui.TextBlock content={label} />
            </gc.ui.Dialog>;
        }

        if (ps === 'complete') {
            let out = job.output ?? {};

            if (!out.url) {
                return <gc.ui.Alert
                    whenClosed={stop}
                    title={this.__('appError')}
                    error={this.__('exporterErrorNoResult')}
                />;
            }

            return <gc.ui.Dialog
                {...gc.lib.cls('exporterResultDialog')}
                title={this.__('exporterComplete')}
                whenClosed={stop}
            >
                <VBox>
                    <VRow flex>
                        <table>
                            <tbody>
                                <tr>
                                    <td>{this.__('exporterNumFeaturesTotal')}</td>
                                    <td>{out['numFeaturesTotal'] ?? 0}</td>
                                </tr>
                                <tr>
                                    <td>{this.__('exporterNumFeaturesExported')}</td>
                                    <td>{out['numFeaturesExported'] ?? 0}</td>
                                </tr>
                                <tr>
                                    <td>{this.__('exporterNumFiles')}</td>
                                    <td>{out['numFiles'] ?? 0}</td>
                                </tr>
                            </tbody>
                        </table>
                    </VRow>

                    <VRow><gc.ui.Button
                        label={this.__('exporterDownload')}
                        primary
                        whenTouched={() => cc.whenDownloadResultButtonsTouched()}

                    />
                    </VRow>
                </VBox>


            </gc.ui.Dialog>;
        }

        if (ps === 'error') {
            return <gc.ui.Alert
                whenClosed={stop}
                title={this.__('appError')}
                error={this.__('exporterError')}
            />;
        }

        return null;
    }
}

class Controller extends gc.Controller {
    uid = MASTER;
    jobTimer: any = null;


    get appOverlayView() {
        return this.createElement(
            this.connect(Dialog, StoreKeys));
    }

    get activeJob() {
        return this.getValue('exportJob');
    }

    async init() {
        await super.init();
        this.app.whenChanged('exportJob', job => this.jobUpdated(job));
        // this.update({ exportState: 'complete' });


    }

    reset() {
        this.update({
            exportJob: null,
            exportState: null,
        });
        clearTimeout(this.jobTimer);
        this.jobTimer = 0;
    }

    stop() {
        this.reset();
    }

    //


    async startJob(res: Promise<gc.gws.JobStatusResponse>) {
        this.update({
            exportJob: { state: gc.gws.JobState.init }
        });

        let job = await res;

        // the job can be canceled in the meantime

        if (!this.activeJob) {
            console.log('JOB ALREADY CANCELED')
            this.sendCancel(job.jobUid);
            return;
        }

        this.update({
            exportJob: job
        });

    }

    async cancelExport() {
        let job = this.activeJob,
            jobUid = job ? job.jobUid : null;

        this.stop();
        await this.sendCancel(jobUid);
    }

    protected jobUpdated(job: gc.gws.JobStatusResponse) {
        if (!job) {
            return this.update({ exportState: null });
        }

        if (job.error) {
            return this.update({ exportState: 'error' });
        }

        console.log('JOB_UPDATED', job.state);

        switch (job.state) {

            case gc.gws.JobState.init:
                this.update({ exportState: 'running' });
                break;

            case gc.gws.JobState.open:
            case gc.gws.JobState.running:
                this.update({ exportState: 'running' });
                this.jobTimer = setTimeout(() => this.poll(), JOB_POLL_INTERVAL);
                break;

            case gc.gws.JobState.cancel:
                this.stop();
                break;

            case gc.gws.JobState.complete:
                gc.lib.nextTick(() => this.update({ exportState: 'complete' }));
                break;

            case gc.gws.JobState.error:
                this.update({ exportState: 'error' });
                break;
        }
    }

    protected async poll() {
        let job = this.getValue('exportJob');

        if (job) {
            this.update({
                exportJob: await this.app.server.call('exporterStatus', { jobUid: job.jobUid }),
            });
        }
    }

    protected async sendCancel(jobUid) {
        if (jobUid) {
            console.log('SEND CANCEL');
            await this.app.server.call('exporterCancel', { jobUid });
        }
    }


    async whenDownloadResultButtonsTouched() {
        let job = this.activeJob;
        if (job && job.output && job.output.url) {
            let u = job.output.url;
            gc.lib.downloadUrl(u, u.split('/').pop(), '_blank');
        }
        this.reset();
    }

}

gc.registerTags({
    [MASTER]: Controller,
});
