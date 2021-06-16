import * as React from 'react';

import * as ui from '../ui';

interface DescriptionProps {
    content: string
}

export class Description extends React.PureComponent<DescriptionProps> {
    render() {
        return <ui.TextBlock
            className='cmpDescription'
            withHTML
            content={this.props.content}
        />
    }
}
