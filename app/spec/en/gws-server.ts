/**
 * Gws Server Base Implementation
 * Version 0.0.8
 */

import * as gws from './gws-server.api';

export abstract class GwsServer implements gws.GwsServerApi {
    abstract async _call(cmd, p): Promise<any>;

    async alkisFsDetails(p: gws.AlkisFsDetailsParams): Promise<gws.AlkisFsDetailsResponse> { return await this._call("alkisFsDetails", p) }
    async alkisFsExport(p: gws.AlkisFsExportParams): Promise<gws.AlkisFsExportResponse> { return await this._call("alkisFsExport", p) }
    async alkisFsPrint(p: gws.AlkisFsPrintParams): Promise<gws.PrinterResponse> { return await this._call("alkisFsPrint", p) }
    async alkisFsSearch(p: gws.AlkisFsQueryParams): Promise<gws.AlkisFsSearchResponse> { return await this._call("alkisFsSearch", p) }
    async alkisFsSetup(p: gws.AlkisFsSetupParams): Promise<gws.AlkisFsSetupResponse> { return await this._call("alkisFsSetup", p) }
    async alkisFsStrassen(p: gws.AlkisFsStrassenParams): Promise<gws.AlkisFsStrassenResponse> { return await this._call("alkisFsStrassen", p) }
    async assetGet(p: gws.AssetParams): Promise<gws.HttpResponse> { return await this._call("assetGet", p) }
    async authCheck(p: gws.NoParams): Promise<gws.AuthResponse> { return await this._call("authCheck", p) }
    async authLogin(p: gws.AuthLoginParams): Promise<gws.AuthResponse> { return await this._call("authLogin", p) }
    async authLogout(p: gws.NoParams): Promise<gws.AuthResponse> { return await this._call("authLogout", p) }
    async editAddFeatures(p: gws.EditParams): Promise<gws.Response> { return await this._call("editAddFeatures", p) }
    async editDeleteFeatures(p: gws.EditParams): Promise<gws.Response> { return await this._call("editDeleteFeatures", p) }
    async editUpdateFeatures(p: gws.EditParams): Promise<gws.Response> { return await this._call("editUpdateFeatures", p) }
    async mapDescribeLayer(p: gws.MapDescribeLayerParams): Promise<gws.HttpResponse> { return await this._call("mapDescribeLayer", p) }
    async mapGetFeatures(p: gws.MapGetFeaturesParams): Promise<gws.MapGetFeaturesResponse> { return await this._call("mapGetFeatures", p) }
    async mapRenderBbox(p: gws.MapRenderBboxParams): Promise<gws.HttpResponse> { return await this._call("mapRenderBbox", p) }
    async mapRenderXyz(p: gws.MapRenderXyzParams): Promise<gws.HttpResponse> { return await this._call("mapRenderXyz", p) }
    async printerCancel(p: gws.PrinterQueryParams): Promise<gws.PrinterResponse> { return await this._call("printerCancel", p) }
    async printerQuery(p: gws.PrinterQueryParams): Promise<gws.PrinterResponse> { return await this._call("printerQuery", p) }
    async printerStart(p: gws.PrintParams): Promise<gws.PrinterResponse> { return await this._call("printerStart", p) }
    async projectInfo(p: gws.ProjectInfoParams): Promise<gws.ProjectInfoResponse> { return await this._call("projectInfo", p) }
    async remoteadminGetSpec(p: gws.RemoteadminGetSpecParams): Promise<gws.Response> { return await this._call("remoteadminGetSpec", p) }
    async remoteadminValidate(p: gws.RemoteadminValidateParams): Promise<gws.Response> { return await this._call("remoteadminValidate", p) }
    async searchFindFeatures(p: gws.SearchParams): Promise<gws.SearchResponse> { return await this._call("searchFindFeatures", p) }
}