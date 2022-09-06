import * as React from 'react';
import * as gws from 'gws';

const {Form, Row, Cell} = gws.ui.Layout;

import * as sidebar from './sidebar';

interface AuthProps extends gws.types.ViewProps {
    controller: SidebarUserTab;
    authLoading: boolean;
    authUsername: string;
    authPassword: string;
    authError?: 'badLogin' | 'badMfa' | 'failedMfa';
    authMfaToken: string;
    authMfaOptions: object;
    user: gws.types.IUser,
}

const StoreKeys = [
    'authLoading',
    'authUsername',
    'authPassword',
    'authError',
    'authMfaToken',
    'authMfaOptions',
];


class UserInfo extends gws.View<AuthProps> {
    render() {
        return <Form>
            <Row>
                <Cell flex>
                    <div className="modUserUserName">{this.props.user.displayName}</div>
                </Cell>
            </Row>
            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.Button
                        primary
                        whenTouched={() => this.props.controller.doLogout()}
                        label={this.__('modUserLogoutButton')}
                    />
                </Cell>
            </Row>
        </Form>;
    }

}

class FormError extends gws.View<AuthProps> {
    render() {
        let msg;

        switch (this.props.authError) {
            case 'badLogin':
                msg = this.__('modUserErrorBadLogin');
                break;
            case 'badMfa':
                msg = this.__('modUserErrorBadMfa');
                break;
            case 'failedMfa':
                msg = this.__('modUserErrorFailedMfa');
                break;
            default:
                return null;
        }

        return <Row>
            <Cell flex>
                <gws.ui.Error text={msg}/>
            </Cell>
        </Row>
    }
}

class LoginForm extends gws.View<AuthProps> {
    render() {
        let submit = () => this.props.controller.doLogin();

        return <Form>
            <FormError {...this.props}/>

            <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        label={this.__('modUserLoginUsername')}
                        value={this.props.authUsername}
                        whenChanged={v => this.props.controller.update({authUsername: v})}
                        whenEntered={submit}
                    />
                </Cell>
            </Row>

            <Row>
                <Cell flex>
                    <gws.ui.PasswordInput
                        label={this.__('modUserLoginPassword')}
                        value={this.props.authPassword}
                        whenChanged={v => this.props.controller.update({authPassword: v})}
                        whenEntered={submit}
                    />
                </Cell>
            </Row>

            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.Button
                        primary
                        whenTouched={submit}
                        label={this.__('modUserLoginButton')}
                    />
                </Cell>
            </Row>


        </Form>;
    }
}

class MfaForm extends gws.View<AuthProps> {
    render() {
        let submit = () => this.props.controller.doMfaVerify();
        let restart = () => this.props.controller.doMfaRestart();

        return <Form>
            <FormError {...this.props}/>

            <Row>
                <Cell flex>
                    <gws.ui.HtmlBlock content={this.props.authMfaOptions['message']}/>
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        value={this.props.authMfaToken}
                        whenChanged={v => this.props.controller.update({authMfaToken: v})}
                        whenEntered={submit}
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.Button
                        primary
                        whenTouched={submit}
                        label={this.__('modUserMfaVerifyButton')}
                    />
                </Cell>
                {this.props.authMfaOptions['allowRestart'] && <Cell>
                    <gws.ui.Button
                        className="modUserMfaRestartButton"
                        whenTouched={restart}
                        tooltip={this.__('modUserMfaRestartButton')}
                    />
                </Cell>}
            </Row>
        </Form>;
    }
}

class SidebarTab extends gws.View<AuthProps> {
    render() {
        let body;

        if (this.props.authLoading)
            body = <gws.ui.Loader/>
        else if (this.props.user)
            body = <UserInfo {...this.props}/>
        else if (this.props.authMfaOptions)
            body = <MfaForm {...this.props}/>
        else
            body = <LoginForm {...this.props}/>;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.props.user
                    ? this.__('modUserInfoTitle')
                    : this.__('modUserLoginTitle')
                }/>
            </sidebar.TabHeader>
            <sidebar.TabBody>{body}</sidebar.TabBody>
        </sidebar.Tab>
    }
}

class SidebarUserTab extends gws.Controller implements gws.types.ISidebarItem {

    iconClass = 'modUserSidebarIcon';

    // async init() {
    //     let res = await this.app.server.authCheck({})
    //
    //     if (res.mfaOptions) {
    //         this.updateAll(null, res.mfaOptions, '', null);
    //         return;
    //     }
    //
    //     this.updateAll(null, null, '', res.user);
    // }

    get tooltip() {
        return this.__('modUserSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarTab, StoreKeys),
            {user: this.getValue('user')}
        );
    }

    async doLogin() {
        this.update({authLoading: true});

        let res = await this.app.server.authLogin({
            username: this.getValue('authUsername') as string,
            password: this.getValue('authPassword') as string,
        });

        if (res.error) {
            this.updateAll('badLogin', null, '', null);
            return;
        }

        if (res.mfaOptions) {
            this.updateAll(null, res.mfaOptions, '', null);
            return;
        }

        this.app.reload();
    }

    async doMfaVerify() {
        this.update({authLoading: true});

        let res = await this.app.server.authMfaVerify({
            token: this.getValue('authMfaToken')
        });

        if (res.error && res.error.status === 409) {
            this.updateAll('badMfa', this.getValue('authMfaOptions'), '', null);
            return;
        }

        if (res.error) {
            this.updateAll('failedMfa', null, '', null);
            return;
        }

        this.app.reload();
    }

    async doMfaRestart() {
        this.update({authLoading: true});

        let res = await this.app.server.authMfaRestart({});

        if (res.error) {
            this.updateAll('failedMfa', null, '', null);
            return;
        }

        if (res.mfaOptions) {
            this.updateAll(null, res.mfaOptions, '', null);
            return;
        }

    }

    async doLogout() {
        this.update({authLoading: true});
        await this.app.server.authLogout({});
        // @TODO control where to redirect in backend
        this.app.navigate('/');
    }

    updateAll(error, opts, token, user) {
        if (user && user.is_guest)
            user = null;

        this.update({
            authLoading: false,
            authError: error,
            authMfaOptions: opts,
            authMfaToken: token,
            user: user,
        });
    }


}

export const tags = {
    'Sidebar.User': SidebarUserTab,
};
