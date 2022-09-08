import * as React from 'react';
import * as gws from 'gws';

const {Form, Row, Cell} = gws.ui.Layout;

import * as sidebar from './sidebar';

interface ComponentProps extends gws.types.ViewProps {
    controller: Controller;
    authLoading: boolean;
    authUsername: string;
    authPassword: string;
    authMfaToken: string;
    label: string;
    user: gws.types.IUser,
}

const ComponentStoreKeys = [
    'authLoading',
    'authUsername',
    'authPassword',
    'authMfaToken',
];

interface TabProps extends gws.types.ViewProps {
    controller: Controller;
    authLoading: boolean;
    user: gws.types.IUser,
}

const TabStoreKeys = [
    'authLoading',
    'authView',
];


class UsernameInput extends gws.View<ComponentProps> {
    render() {
        return <gws.ui.TextInput
            label={this.props.label}
            value={this.props.authUsername}
            whenChanged={v => this.props.controller.update({authUsername: v})}
            whenEntered={() => this.props.controller.doLogin()}
        />

    }
}

class PasswordInput extends gws.View<ComponentProps> {
    render() {
        return <gws.ui.PasswordInput
            label={this.props.label}
            value={this.props.authPassword}
            whenChanged={v => this.props.controller.update({authPassword: v})}
            whenEntered={() => this.props.controller.doLogin()}
        />
    }
}

class LoginButton extends gws.View<ComponentProps> {
    render() {
        return <gws.ui.Button
            label={this.props.label}
            primary
            whenTouched={() => this.props.controller.doLogin()}
        />
    }
}

class LogoutButton extends gws.View<ComponentProps> {
    render() {
        return <gws.ui.Button
            label={this.props.label}
            primary
            whenTouched={() => this.props.controller.doLogout()}
        />
    }
}

class MfaInput extends gws.View<ComponentProps> {
    render() {
        return <gws.ui.TextInput
            label={this.props.label}
            value={this.props.authMfaToken}
            whenChanged={v => this.props.controller.update({authMfaToken: v})}
            whenEntered={() => this.props.controller.doMfaVerify()}
        />
    }
}

class MfaVerifyButton extends gws.View<ComponentProps> {
    render() {
        return <gws.ui.Button
            label={this.props.label}
            primary
            whenTouched={() => this.props.controller.doMfaVerify()}
        />

    }
}

class MfaRestartButton extends gws.View<ComponentProps> {
    render() {
        return <gws.ui.Button
            tooltip={this.props.label}
            className="modAuthMfaRestartButton"
            whenTouched={() => this.props.controller.doMfaRestart()}
        />
    }
}

class SidebarTab extends gws.View<TabProps> {
    body() {

        return ;
    }

    render() {
        let body = this.props.controller.renderSidebarBody();

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.props.controller.isLoggedIn()
                    ? this.__('modUserInfoTitle')
                    : this.__('modUserLoginTitle')
                }/>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                {body}
                {this.props.authLoading && <gws.ui.Loader/>}
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}


class Controller extends gws.Controller implements gws.types.ISidebarItem {

    iconClass = 'modAuthSidebarIcon';

    tags = {
        'Auth.UsernameInput': UsernameInput,
        'Auth.PasswordInput': PasswordInput,
        'Auth.LoginButton': LoginButton,
        'Auth.LogoutButton': LogoutButton,
        'Auth.MfaInput': MfaInput,
        'Auth.MfaVerifyButton': MfaVerifyButton,
        'Auth.MfaRestartButton': MfaRestartButton,
        'Box': Form,
        'Cell': Cell,
        'Row': Row,
        'Error': gws.ui.Error,
    };

    jsxProps = {
        'classname': 'className',
        'class': 'className',
    };


    async init() {
        let res = await this.app.server.authCheck({})
        this.updateFromResult(res)
    }


    renderSidebarBody() {
        let tpl = this.getValue('authView');
        if (tpl)
            return this.renderTemplate(tpl);
        return null;
    }

    makeElement(el: Node) {
        let children = [];

        for (let node of Array.from(el.childNodes)) {
            let sub = (node.nodeType === node.TEXT_NODE)
                ? node.textContent.trim()
                : this.makeElement(node);
            if (sub)
                children.push(sub);
        }

        let props = {};

        for (let attr of el['attributes']) {
            let a = attr.name.toLowerCase();
            let v = attr.value;

            if (v === 'true' || v === '')
                v = true;
            if (v === 'false')
                v = false;

            props[this.jsxProps[a] || a] = v;
        }

        let tagName = el['tagName'];

        console.log('>>>>', el['tagName'], tagName, props, children)

        if (this.tags[tagName] && tagName.includes('.')) {
            return this.createElement(
                this.connect(this.tags[tagName], ComponentStoreKeys),
                {user: this.getValue('user'), ...props}
            );
        }

        let cls = this.tags[tagName] || tagName.toLowerCase();
        return React.createElement(cls, props, ...children);

    }

    renderTemplate(txt) {
        let parser = new DOMParser();
        let doc = parser.parseFromString(txt, 'text/xml');
        return this.makeElement(doc.firstChild);
    }


    get tooltip() {
        return this.__('modAuthSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarTab, TabStoreKeys),
            {user: this.getValue('user')}
        );
    }

    async doLogin() {
        this.update({authLoading: true});

        let res = await this.app.server.authLogin({
            username: this.getValue('authUsername') as string,
            password: this.getValue('authPassword') as string,
        });

        this.updateFromResult(res);
    }

    async doMfaVerify() {
        this.update({authLoading: true});

        let res = await this.app.server.authMfaVerify({
            token: this.getValue('authMfaToken')
        });

        this.updateFromResult(res);
    }

    async doMfaRestart() {
        this.update({authLoading: true});
        let res = await this.app.server.authMfaRestart({});
        this.updateFromResult(res);
    }

    async doLogout() {
        this.update({authLoading: true});
        await this.app.server.authLogout({});
        // @TODO control where to redirect in backend
        this.app.navigate('/');
    }

    isLoggedIn() {
        let user = this.getValue('user');
        return user && !user.isGuest;
    }

    updateFromResult(res) {
        if (res.actionResult === 'loginOk') {
            this.app.reload();
            return;
        }

        let user = res.user;

        if (user && user.isGuest)
            user = null;

        this.update({
            authView: res.view,
            authLoading: false,
            authMfaToken: '',
            user: user,
        });
    }


}

export const tags = {
    'Sidebar.Auth': Controller,
};
