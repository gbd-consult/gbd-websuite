import * as React from 'react';
import * as gws from 'gws';

import * as sidebar from './sidebar';

interface AuthProps extends gws.types.ViewProps {
    controller: SidebarUserTab;
    authUsername: string;
    authPassword: string;
    authError: boolean;
    user: gws.types.IUser,
}

class Icon extends gws.View<AuthProps> {
    render() {
        return <div>A</div>;
    }
}

class UserInfo extends gws.View<AuthProps> {
    render() {
        let {Form, Row, Cell} = gws.ui.Layout;

        return <Form>
            <Row>
                <Cell flex>
                    <div className="modUserUserName">{this.props.user.displayName}</div>
                </Cell>
            </Row>
            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.TextButton
                        primary
                        whenTouched={() => this.props.controller.doLogout()}
                    >{this.__('modUserLogoutButton')}</gws.ui.TextButton>
                </Cell>
            </Row>
        </Form>;
    }

}

class LoginForm extends gws.View<AuthProps> {
    render() {
        let submit = () => this.props.controller.doLogin();
        let {Form, Row, Cell} = gws.ui.Layout;

        return <Form>
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
                    <gws.ui.TextButton
                        primary
                        whenTouched={submit}
                    >{this.__('modUserLoginButton')}</gws.ui.TextButton>

                </Cell>
            </Row>

            {this.props.authError && <Row>
                <Cell flex>
                    <gws.ui.Error text={this.__('modUserError')}/>
                </Cell>
            </Row>}

        </Form>;
    }
}

class SidebarTab extends gws.View<AuthProps> {
    render() {
        let {Row, Cell} = gws.ui.Layout;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.props.user
                        ? this.__('modUserInfoTitle')
                        : this.__('modUserLoginTitle')
                    }/>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                {this.props.user
                    ? <UserInfo {...this.props}/>
                    : <LoginForm {...this.props}/>
                }
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

class SidebarUserTab extends gws.Controller implements gws.types.ISidebarItem {

    async doLogin() {
        let res = await this.app.server.authLogin({
            username: this.getValue('authUsername') as string,
            password: this.getValue('authPassword') as string,
        });

        if (res.error)
            this.update({authError: true});
        else
        //this.update({user: res.user, authError: false});
            this.app.reload();
    }

    async doLogout() {
        await this.app.server.authLogout({});
        this.app.reload();
    }

    get iconClass() {
        return 'modUserSidebarIcon';
    }

    get tooltip() {
        return this.__('modUserTooltip');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarTab, ['authUsername', 'authPassword', 'authError']),
            {user: this.getValue('user')}
        );
    }

}

export const tags = {
    'Sidebar.User': SidebarUserTab,
};
