import * as React from 'react';

import * as gc from 'gc';
;
import * as sidebar from 'gc/elements/sidebar';

const {Form, Row, Cell} = gc.ui.Layout;

interface MfaProps {
    code: string;
    message: string;
    canRestart: boolean;
    error: boolean;
}

interface ViewProps extends gc.types.ViewProps {
    controller: SidebarUserTab;

    authLoading: boolean;
    authUsername: string;
    authPassword: string;
    authError: boolean;
    authMfa: MfaProps | null;
    user: gc.types.IUser | null;
}

const StoreKeys = [
    'authLoading',
    'authUsername',
    'authPassword',
    'authError',
    'authMfa',
];


class UserInfo extends gc.View<ViewProps> {
    render() {
        return <Form>
            <Row>
                <Cell flex>
                    <div className="userUserName">{this.props.user.displayName}</div>
                </Cell>
            </Row>
            <Row>
                <Cell flex/>
                <Cell>
                    <gc.ui.Button
                        primary
                        whenTouched={() => this.props.controller.whenLogoutFormSubmitted()}
                        label={this.__('userLogoutButton')}
                    />
                </Cell>
            </Row>
        </Form>;
    }

}

class LoginForm extends gc.View<ViewProps> {
    render() {
        let submit = () => this.props.controller.whenLoginFormSubmitted();

        return <Form>
            <Row>
                <Cell flex>
                    <gc.ui.TextInput
                        label={this.__('userLoginUsername')}
                        value={this.props.authUsername}
                        whenChanged={v => this.props.controller.update({authUsername: v})}
                        whenEntered={submit}
                    />
                </Cell>
            </Row>

            <Row>
                <Cell flex>
                    <gc.ui.PasswordInput
                        label={this.__('userLoginPassword')}
                        value={this.props.authPassword}
                        whenChanged={v => this.props.controller.update({authPassword: v})}
                        whenEntered={submit}
                    />
                </Cell>
            </Row>

            <Row>
                <Cell flex/>
                <Cell>
                    <gc.ui.Button
                        primary
                        whenTouched={submit}
                        label={this.__('userLoginButton')}
                    />
                </Cell>
            </Row>

            {this.props.authError && <Row>
                <Cell flex>
                    <gc.ui.Error text={this.__('userError')}/>
                </Cell>
            </Row>}

        </Form>;
    }
}

class MfaForm extends gc.View<ViewProps> {
    render() {
        let submit = () => this.props.controller.whenMfaFormSubmitted();
        let restart = () => this.props.controller.whenMfaRestartButtonTouched();
        let update = (v) => this.props.controller.update({authMfa: {...mfa, code: v}})

        let mfa = this.props.authMfa;

        return <Form>
            <Row>
                <Cell flex>
                    <gc.ui.HtmlBlock content={mfa.message}/>
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    <gc.ui.TextInput
                        value={mfa.code}
                        whenChanged={update}
                        whenEntered={submit}
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex/>
                <Cell>
                    <gc.ui.Button
                        primary
                        whenTouched={submit}
                        label={this.__('userMfaVerifyButton')}
                    />
                </Cell>
                {mfa.canRestart && <Cell>
                    <gc.ui.Button
                        whenTouched={restart}
                        label={this.__('userMfaRestartButton')}
                    />
                </Cell>}
            </Row>
            {mfa.error && <Row>
                <Cell flex>
                    <gc.ui.Error text={this.__('userMfaError')}/>
                </Cell>
            </Row>}
        </Form>;
    }
}


class SidebarTab extends gc.View<ViewProps> {
    render() {
        let title = '', body;

        if (0 && this.props.authLoading) {
            title = '';
            body = <gc.ui.Loader/>;
        } else if (this.props.authMfa) {
            title = this.__('userMfaTitle');
            body = <MfaForm {...this.props}/>;
        } else if (this.props.user) {
            title = this.__('userInfoTitle');
            body = <UserInfo {...this.props}/>
        } else {
            title = this.__('userLoginTitle');
            body = <LoginForm {...this.props}/>;
        }

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gc.ui.Title content={title}/>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                {body}
                {this.props.authLoading && <gc.ui.Loader/>}
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

class SidebarUserTab extends gc.Controller implements gc.types.ISidebarItem {

    iconClass = 'userSidebarIcon';

    get tooltip() {
        return this.__('userSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarTab, StoreKeys),
            {user: this.getValue('user')}
        );
    }

    async whenLoginFormSubmitted() {
        this.update({
            authLoading: true,
        });

        let res = await this.app.server.call('authLogin', {
            username: this.getValue('authUsername') as string,
            password: this.getValue('authPassword') as string,
        });

        if (res.error) {
            return this.setError();
        }

        if (res.mfaState) {
            return this.setMfa(res, false);
        }

        this.whenLoggedIn(res.user);
    }

    async whenMfaFormSubmitted() {
        this.update({
            authLoading: true,
        });

        let mfa = this.getValue('authMfa') as MfaProps;
        let res = await this.app.server.call('authMfaVerify', {
            payload: {code: mfa.code},
        });

        if (res.error) {
            return this.setError();
        }

        if (res.mfaState === gc.gws.AuthMultiFactorState.retry) {
            return this.setMfa(res, true);
        }

        if (res.mfaState === gc.gws.AuthMultiFactorState.ok) {
            this.whenLoggedIn(res.user);
            return;
        }

        this.setError();
    }

    async whenMfaRestartButtonTouched() {
        this.update({
            authLoading: true,
        });

        let mfa = this.getValue('authMfa') as MfaProps;
        let res = await this.app.server.call('authMfaRestart', {});

        if (res.error) {
            return this.setError();
        }

        return this.setMfa(res, false);
    }

    async whenLogoutFormSubmitted() {
        await this.app.server.call('authLogout', {});
        this.whenLoggedOut()
    }

    whenLoggedIn(user) {
        this.app.reload();
    }

    whenLoggedOut() {
        // @TODO control where to redirect in backend
        this.app.navigate('/');
    }

    setError() {
        this.update({
            authLoading: false,
            authError: true,
            authMfa: null,
            user: null,
        });
    }

    setMfa(res: gc.gws.plugin.auth_method.web.LoginResponse, error: boolean) {
        this.update({
            authLoading: false,
            authError: false,
            authMfa: {
                code: '',
                message: res.mfaMessage,
                canRestart: res.mfaCanRestart,
                error,
            },
        });
    }


}

gc.registerTags({
    'Sidebar.User': SidebarUserTab,
});
