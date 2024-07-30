import * as React from 'react';

import * as gws from 'gws';

let {Form, Row, Cell} = gws.ui.Layout;


const MASTER = 'Shared.Account';

class AccountState {
    page: string;
    tc: string;
    onboardingEmail: string;
    newPassword1: string;
    newPassword2: string;
    mfa: Array<gws.api.plugin.account.account_action.MfaProps>
    mfaIndex: number
    error: string;
}

interface ViewProps extends gws.types.ViewProps {
    controller: Controller;
    accountState: AccountState;
}


const StoreKeys = [
    'accountState',
];

class PageOnboardingPassword extends gws.View<ViewProps> {
    render() {
        let cc = this.props.controller;
        let es = this.props.accountState;

        return <Form>
            <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        label={this.__('onboardingEmail')}
                        value={es.onboardingEmail}
                        whenChanged={v => cc.whenOnboardingEmailChanged(v)}
                        whenEntered={() => cc.whenSubmitted()}
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    <gws.ui.PasswordInput
                        label={this.__('newPassword1')}
                        value={es.newPassword1}
                        withShow
                        whenChanged={v => cc.whenNewPasswordChanged(v, 'newPassword1')}
                        whenEntered={() => cc.whenSubmitted()}
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    <gws.ui.PasswordInput
                        label={this.__('newPassword2')}
                        value={es.newPassword2}
                        withShow
                        whenChanged={v => cc.whenNewPasswordChanged(v, 'newPassword2')}
                        whenEntered={() => cc.whenSubmitted()}
                    />
                </Cell>
            </Row>
            {es.error && <Row>
                <Cell flex>
                    <gws.ui.Error text={es.error}/>
                </Cell>
            </Row>}
        </Form>
    }
}

class PageOnboardingMfa extends gws.View<ViewProps> {
    render() {
        let cc = this.props.controller;
        let es = this.props.accountState;

        let rows = [];

        for (let mfa of es.mfa) {
            rows.push(
                <Row>
                    <Cell flex>
                        <gws.ui.Toggle
                            type="radio"
                            label={mfa.title}
                            inline
                            value={es.mfaIndex === mfa.index}
                            whenChanged={v => cc.whenMfaSelected(mfa.index)}
                        />
                    </Cell>
                </Row>
            )
            if (mfa.qrCode) {
                rows.push(
                    <Row>
                        <Cell flex>
                            <img className="accountQrCode" src={mfa.qrCode}/>
                        </Cell>
                    </Row>
                )
            }
        }

        return <Form>
            {rows}
        </Form>
    }
}

class PageOnboardingComplete extends gws.View<ViewProps> {
    render() {
        return <Form>
            <Row>
                <Cell flex>
                    <gws.ui.TextBlock
                        content={this.__('textOnboardingComplete')}
                    />
                </Cell>
            </Row>
        </Form>
    }
}

class PageFatalError extends gws.View<ViewProps> {
    render() {
        let es = this.props.accountState;

        return <Form>
            {es.error && <Row>
                <Cell flex>
                    <gws.ui.Error text={es.error}/>
                </Cell>
            </Row>}
        </Form>
    }
}

//

class DialogContent extends gws.View<ViewProps> {
    render() {
        let es = this.props.accountState;

        switch (es.page) {
            case 'OnboardingPassword':
                return <PageOnboardingPassword {...this.props}/>
            case 'OnboardingMfa':
                return <PageOnboardingMfa {...this.props}/>
            case 'OnboardingComplete':
                return <PageOnboardingComplete {...this.props}/>
            case 'FatalError':
                return <PageFatalError {...this.props}/>
        }
    }
}

class Dialog extends gws.View<ViewProps> {
    render() {
        let cc = this.props.controller;
        let es = this.props.accountState;

        if (!es.page) {
            return null
        }

        let title = cc.dialogTitle[es.page];

        let okButton = <gws.ui.Button
            {...gws.lib.cls('editSaveButton', 'isActive')}
            tooltip={this.__('editSave')}
            disabled={!cc.canSubmit()}
            whenTouched={() => cc.whenSubmitted()}
        />

        return <gws.ui.Dialog
            className="accountDialog"
            title={title}
            buttons={[okButton]}
            whenClosed={() => cc.closeDialog()}
        >
            <DialogContent {...this.props}/>
        </gws.ui.Dialog>;
    }

}

//

class Controller extends gws.Controller {
    uid = MASTER;
    dialogTitle: object

    async init() {
        await super.init();

        this.updateState({
            page: '',
        });

        this.dialogTitle = {
            'OnboardingPassword': this.__('titleOnboardingPassword'),
            'OnboardingMfa': this.__('titleOnboardingMfa'),
            'OnboardingComplete': this.__('titleOnboardingComplete'),
            'FatalError': this.__('titleFatalError'),
        }

        this.app.whenLoaded(() => this.whenAppLoaded());
    }

    async whenAppLoaded() {
        let onboardingTc = this.app.urlParams['onboarding'];

        if (onboardingTc) {
            await this.startOnboarding(onboardingTc)
        }
    }


    get appOverlayView() {
        return this.createElement(
            this.connect(Dialog, StoreKeys));
    }

    //

    async startOnboarding(tc) {
        let res = await this.app.server.accountOnboardingStart({tc: tc})

        if (res.error) {
            return this.updateState({
                page: 'FatalError',
            });
        }
        this.updateState({
            page: 'OnboardingPassword',
            tc: res.tc,
        });
    }

    whenOnboardingEmailChanged(v) {
        this.updateState({
            onboardingEmail: v
        })
    }

    whenNewPasswordChanged(v, key) {
        this.updateState({
            [key]: v.trim()
        });
        this.updateState({
            error: this.validateNewPassword()
        })
    }

    validateNewPassword() {
        let es = this.accountState;
        let p1 = es.newPassword1;
        let p2 = es.newPassword2;

        if (gws.lib.isEmpty(p1) || gws.lib.isEmpty(p2)) {
            return this.__('errorEmptyPassword')
        }
        if (p1 !== p2) {
            return this.__('errorPasswordsDontMatch')
        }
        return '';
    }

    whenMfaSelected(index) {
        this.updateState({
            mfaIndex: index
        })
    }

    async whenSubmitted() {
        if (!this.canSubmit()) {
            return;
        }

        let es = this.accountState;

        switch (es.page) {
            case 'OnboardingPassword': {
                let res = await this.app.server.accountOnboardingSavePassword({
                    tc: es.tc,
                    email: es.onboardingEmail,
                    password1: es.newPassword1,
                    password2: es.newPassword2,
                })
                if (res.error) {
                    return this.updateState({
                        error: this.__('serverError'),
                    });
                }
                if (res.complete) {
                    return this.updateState({
                        page: 'OnboardingComplete'
                    })
                }
                return this.updateState({
                    page: 'OnboardingMfa',
                    tc: res.tc,
                    mfa: res.mfa,
                })
            }
            case 'OnboardingMfa': {
                let res = await this.app.server.accountOnboardingSaveMfa({
                    tc: es.tc,
                    mfaIndex: es.mfaIndex,
                })
                if (res.error) {
                    return this.updateState({
                        error: this.__('serverError'),
                    });
                }
                if (res.complete) {
                    return this.updateState({
                        page: 'OnboardingComplete'
                    })
                }
            }
        }
    }

    //

    canSubmit() {
        let es = this.accountState;

        switch (es.page) {
            case 'OnboardingPassword':
                if (gws.lib.isEmpty(es.onboardingEmail)) {
                    return false;
                }
                return this.validateNewPassword() === ''
            case 'OnboardingMfa':
                return !!es.mfaIndex;
        }

        return true
    }


    closeDialog() {
        this.updateState({page: ''});
    }

    get accountState(): AccountState {
        return this.getValue('accountState') || {};
    }

    updateState(es: Partial<AccountState> = null) {
        let s = {
            ...this.accountState,
            ...(es || {})
        }

        this.update({
            accountState: s
        });
    }
}


gws.registerTags({
    [MASTER]: Controller,
    'Account.Dialog': Dialog,
});
