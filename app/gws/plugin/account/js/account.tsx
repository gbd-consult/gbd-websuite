import * as React from 'react';

import * as gc from 'gc';
;

let {Form, Row, Cell} = gc.ui.Layout;


const MASTER = 'Shared.Account';

class AccountState {
    page: string;
    completionUrl: string;
    tc: string;
    onboardingEmail: string;
    newPassword1: string;
    newPassword2: string;
    mfaList: Array<gc.gws.plugin.account.account_action.MfaProps>
    mfaIndex: number
    errorText: string;
}

interface ViewProps extends gc.types.ViewProps {
    controller: Controller;
    accountState: AccountState;
}


const StoreKeys = [
    'accountState',
];

class PageOnboardingPassword extends gc.View<ViewProps> {
    renderForm() {
        let cc = this.props.controller;

        return <Form>
            <Row>
                <Cell flex>
                    <gc.ui.TextBlock content={this.__('accountOnboardingPasswordText')}/>
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    <gc.ui.TextInput
                        label={this.__('accountOnboardingEmail')}
                        value={cc.accountState.onboardingEmail}
                        whenChanged={v => cc.whenOnboardingEmailChanged(v)}
                        whenEntered={() => cc.whenOnboardingPasswordConfirmed()}
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    <gc.ui.PasswordInput
                        label={this.__('accountNewPassword1')}
                        value={cc.accountState.newPassword1}
                        withShow
                        whenChanged={v => cc.whenNewPasswordChanged(v, 'newPassword1')}
                        whenEntered={() => cc.whenOnboardingPasswordConfirmed()}
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    <gc.ui.PasswordInput
                        label={this.__('accountNewPassword2')}
                        value={cc.accountState.newPassword2}
                        withShow
                        whenChanged={v => cc.whenNewPasswordChanged(v, 'newPassword2')}
                        whenEntered={() => cc.whenOnboardingPasswordConfirmed()}
                    />
                </Cell>
            </Row>
            {cc.accountState.errorText && <Row>
                <Cell flex>
                    <gc.ui.Error text={cc.accountState.errorText}/>
                </Cell>
            </Row>}
        </Form>
    }

    render() {
        let cc = this.props.controller;
        let buttonEnabled = true;

        if (gc.lib.isEmpty(cc.accountState.onboardingEmail)) {
            buttonEnabled = false;
        } else {
            buttonEnabled = cc.validateNewPassword() === '';
        }

        let okButton = <gc.ui.Button
            {...gc.lib.cls('editSaveButton', 'isActive')}
            tooltip={this.__('accountSave')}
            disabled={!buttonEnabled}
            whenTouched={() => cc.whenOnboardingPasswordConfirmed()}
        />

        return <gc.ui.Dialog
            className="accountDialog"
            title={this.__('accountOnboardingTitle')}
            buttons={[okButton]}
        >
            {this.renderForm()}
        </gc.ui.Dialog>;
    }

}

class PageOnboardingMfa extends gc.View<ViewProps> {
    renderForm() {
        let cc = this.props.controller;
        let rows = [];

        rows.push(
            <Row>
                <Cell flex>
                    <gc.ui.Text content={this.__('accountOnboardingMfaText')}/>
                </Cell>
            </Row>
        )

        for (let mfa of cc.accountState.mfaList) {
            rows.push(
                <Row>
                    <Cell flex>
                        <gc.ui.Toggle
                            type="radio"
                            label={mfa.title}
                            inline
                            value={cc.accountState.mfaIndex === mfa.index}
                            whenChanged={v => cc.whenMfaChanged(mfa.index)}
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

    render() {
        let cc = this.props.controller;
        let buttonEnabled = !!cc.accountState.mfaIndex;

        let okButton = <gc.ui.Button
            {...gc.lib.cls('editSaveButton', 'isActive')}
            tooltip={this.__('accountSave')}
            disabled={!buttonEnabled}
            whenTouched={() => cc.whenOnboardingMfaConfirmed()}
        />

        return <gc.ui.Dialog
            className="accountDialog"
            title={this.__('accountOnboardingTitle')}
            buttons={[okButton]}
        >
            {this.renderForm()}
        </gc.ui.Dialog>;
    }
}

class PageOnboardingComplete extends gc.View<ViewProps> {
    render() {
        let cc = this.props.controller;

        let okButton = <gc.ui.Button
            {...gc.lib.cls('editSaveButton', 'isActive')}
            whenTouched={() => cc.whenOnboardingCompleteConfirmed()}
        />

        return <gc.ui.Dialog
            className="accountDialog"
            title={this.__('accountOnboardingTitle')}
            buttons={[okButton]}
        >
            <gc.ui.Info text={this.__('accountOnboardingComplete')}/>
        </gc.ui.Dialog>;
    }
}

class PageFatalError extends gc.View<ViewProps> {
    render() {
        let cc = this.props.controller;
        let es = this.props.accountState;

        return <gc.ui.Alert
            title={this.__('accountError')}
            error={es.errorText}
            whenClosed={() => cc.closeDialog()}
        />
    }
}

//

class Dialog extends gc.View<ViewProps> {
    render() {
        let es = this.props.accountState;

        if (!es.page) {
            return null
        }

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

//

class Controller extends gc.Controller {
    uid = MASTER;

    async init() {
        await super.init();

        this.updateState({
            page: '',
        });

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
            return this.showOnboardingError();
        }
        this.updateState({
            page: 'OnboardingPassword',
            tc: res.tc,
        });
    }

    showOnboardingError() {
        return this.updateState({
            page: 'FatalError',
            errorText: this.__('accountOnboardingError'),
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
            errorText: this.validateNewPassword()
        })
    }

    validateNewPassword() {
        let es = this.accountState;
        let p1 = es.newPassword1;
        let p2 = es.newPassword2;

        if (gc.lib.isEmpty(p1)) {
            return this.__('accountErrorEmptyPassword')
        }
        if (p1 !== p2) {
            return this.__('accountErrorPasswordsDontMatch')
        }
        return '';
    }

    whenMfaChanged(index) {
        this.updateState({
            mfaIndex: index
        })
    }

    //

    async whenOnboardingPasswordConfirmed() {
        let es = this.accountState;

        if (gc.lib.isEmpty(es.onboardingEmail)) {
            return this.updateState({errorText: this.__('accountErrorEmptyEmail')})
        }

        let err = this.validateNewPassword();
        if (err) {
            return this.updateState({errorText: err})
        }

        let res = await this.app.server.accountOnboardingSavePassword({
            tc: es.tc,
            email: es.onboardingEmail,
            password1: es.newPassword1,
            password2: es.newPassword2,
        })
        if (res.error) {
            return this.showOnboardingError()
        }
        if (res.complete) {
            return this.updateState({
                page: 'OnboardingComplete',
                completionUrl: res.completionUrl,
            })
        }
        if (!res.ok) {
            return this.updateState({
                errorText: this.__('accountOnboardingError'),
                tc: res.tc,
            })
        }
        return this.updateState({
            page: 'OnboardingMfa',
            tc: res.tc,
            mfaList: res.mfaList,
        })
    }

    async whenOnboardingMfaConfirmed() {
        let es = this.accountState;

        let res = await this.app.server.accountOnboardingSaveMfa({
            tc: es.tc,
            mfaIndex: es.mfaIndex,
        })
        if (res.error) {
            return this.showOnboardingError()
        }
        if (res.complete) {
            return this.updateState({
                page: 'OnboardingComplete',
                completionUrl: res.completionUrl,
            })
        }

    }

    async whenOnboardingCompleteConfirmed() {
        let es = this.accountState;
        this.app.navigate(es.completionUrl);
    }

    //

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


gc.registerTags({
    [MASTER]: Controller,
    'Account.Dialog': Dialog,
});
