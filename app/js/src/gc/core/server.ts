import axios, {Canceler, ResponseType} from 'axios';
import * as msgpack from '@ygoe/msgpack';

import {BaseServer} from '../gws';

function errorResponse(err) {
    console.log('AXIOS ERROR', err);
    return {
        error: {
            status: err.response ? err.response.status : -1,
            info: String(err)
        }
    }
}

class LayerLoader {
    cancel: Canceler;
    promise: Promise<any>;

    constructor(url: string, responseType: ResponseType, whenDone: (ldr: LayerLoader) => void) {
        this.promise = this.load(url, responseType, whenDone);
    }

    async load(url: string, responseType: ResponseType, whenDone: (ldr: LayerLoader) => void) {
        let data = null;
        try {
            let res = await axios.get(url, {
                cancelToken: new axios.CancelToken(c => this.cancel = c),
                responseType,
            });
            data = res.data;
        } catch (err) {
            if (!axios.isCancel(err))
                console.log('LOAD ERROR', url, err);
        }
        whenDone(this);
        return data;
    }
}

const MSGPACK_MIME = 'application/msgpack';

export class Server extends BaseServer {
    url: string;
    whenChanged: () => void;
    commandCount = 0;
    activeLoaders: {[layerUid: string]: LayerLoader} = {};

    protected app;

    constructor(app, url) {
        super();
        this.url = url;
        this.whenChanged = () => null;
        this.app = app;
    }

    get requestCount() {
        return this.commandCount + Object.keys(this.activeLoaders).length;
    }

    async execCall(cmd, request, options) {
        request = request || {};
        request.projectUid = request.projectUid || this.app.project?.uid;
        request.localeUid = request.localeUid || this.app.localeUid;

        this.commandCount++;
        this.whenChanged();
        let res = await this.execCall2(cmd, request, options);
        this.commandCount--;
        this.whenChanged();
        return res;
    }

    async execCall2(cmd, request, options) {
        let req: any = {
            url: this.url + '/' + cmd,
            method: 'POST',
            data: request,
            withCredentials: true
        };

        let binaryRequest = false;
        let binaryResponse = false;

        if (options) {
            binaryRequest = !!options.binaryRequest;
            binaryResponse = !!options.binaryResponse;
            if (options.binary)
                binaryRequest = binaryResponse = true;
        }

        req.headers = {}

        if (binaryRequest) {
            req.data = msgpack.serialize(req.data);
            req.headers['content-type'] = MSGPACK_MIME
        }
        if (binaryResponse) {
            req.headers['accept'] = MSGPACK_MIME;
            req.responseType = 'arraybuffer';
        }

        try {
            let res = await axios.request(req),
                data = res.data;
            if (res.headers['content-type'].includes(MSGPACK_MIME))
                data = msgpack.deserialize(new Uint8Array(data));
            return data;
        } catch (err) {
            if (binaryResponse)
                return errorResponse(err);
            if (err && err.response && err.response.data)
                return err.response.data;
            return errorResponse(err);
        }
    }

    loadLayerData(layerUid, url, responseType) {
        this.cancelLayerLoad(layerUid);

        if (!url)
            return;

        let whenDone = (ldr: LayerLoader) => {
            if (this.activeLoaders[layerUid] === ldr) {
                delete this.activeLoaders[layerUid];
                this.whenChanged();
            }
        };
        let loader = new LayerLoader(url, responseType, whenDone);
        this.activeLoaders[layerUid] = loader;
        this.whenChanged();

        return loader.promise;
    }

    cancelLayerLoad(layerUid) {
        let loader = this.activeLoaders[layerUid];
        if (!loader)
            return;
        delete this.activeLoaders[layerUid];
        loader.cancel();
        this.whenChanged();
    }
}
