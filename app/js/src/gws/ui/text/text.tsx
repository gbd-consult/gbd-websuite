import * as React from 'react';

import * as base from './base';


interface ErrorProps {
    text: string;
    details?: string;
}

export class Error extends base.Pure<ErrorProps> {
    render() {
        return <div className='uiError'>
            <div className="uiErrorText">{this.props.text}</div>
            {this.props.details && <div className="uiErrorDetails">{this.props.details}</div>}
        </div>
    }
}

//

interface InfoProps {
    text: string;
    details?: string;
}

export class Info extends base.Pure<InfoProps> {
    render() {
        return <div className='uiInfo'>
            <div className="uiInfoText">{this.props.text}</div>
            {this.props.details && <div className="uiInfoDetails">{this.props.details}</div>}
        </div>
    }
}

//

interface TextProps {
    content: string;
    whenTouched?: () => any;
    className?: string;
}


export class Title extends base.Pure<TextProps> {
    render() {
        let cls = this.props.className || 'uiTitle';
        return <Text {...this.props} className={cls}/>;
    }
}

export class Text extends base.Pure<TextProps> {
    render() {
        return <div
            className={this.props.className || 'uiText'}
            onClick={this.props.whenTouched}
        >{this.props.content}</div>;
    }
}

//

interface LinkProps extends TextProps {
    href?: string;
    target?: string;
}

export class Link extends base.Pure<LinkProps> {
    render() {
        let cls = this.props.className || 'uiLink';

        if (this.props.whenTouched) {
            return <a className={cls} onClick={() => this.props.whenTouched()}>{this.props.content}</a>;
        }

        if (this.props.href) {
            return <a className={cls} href={this.props.href} target={this.props.target}>{this.props.content}</a>;
        }
        return <a className={cls}>{this.props.content}</a>;
    }
}

//

interface TextBlockProps {
    content: string;
    withHTML?: boolean;
    className?: string;
    whenTouched?: (e: React.MouseEvent) => void;
}

export class TextBlock extends base.Pure<TextBlockProps> {

    render() {
        let s = String(this.props.content || '').trim();
        if (!s)
            return null;

        let cls = this.props.className || 'uiTextBlock';
        let c;

        c = (s[0] === '<' && this.props.withHTML) ? asHtml(s) : s;
        return <div
            className={cls}
            onClick={this.props.whenTouched}
        >{c}</div>;
    }
}

//

interface HtmlBlockProps {
    content: string;
    className?: string;
}

export class HtmlBlock extends base.Pure<HtmlBlockProps> {
    render() {
        let s = String(this.props.content || '').trim();
        if (!s)
            return null;
        return <div
            className={this.props.className || 'uiTextBlock'}
        >{asHtml(s)}</div>;
    }
}

function asHtml(s) {
    return <div dangerouslySetInnerHTML={{__html: s}}/>;
}
