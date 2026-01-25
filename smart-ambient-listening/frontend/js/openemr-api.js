/**
 * OpenEMR REST API Client
 *
 * Provides access to OpenEMR's standard REST API endpoints,
 * including custom user settings, using the SMART on FHIR access token.
 *
 * Usage:
 *   // Initialize with SMART client
 *   const smartClient = await FHIR.oauth2.ready();
 *   const openemrApi = new OpenEMRApi(smartClient);
 *
 *   // Get custom user settings
 *   const settings = await openemrApi.getCustomUserSettings();
 *
 *   // Get available fields
 *   const fields = await openemrApi.getCustomUserSettingsFields();
 *
 *   // Get a specific setting
 *   const setting = await openemrApi.getCustomUserSetting('my_field_id');
 *
 *   // Update a setting
 *   await openemrApi.updateCustomUserSetting('my_field_id', 'new_value');
 *
 *   // Delete/reset a setting
 *   await openemrApi.deleteCustomUserSetting('my_field_id');
 */

class OpenEMRApi {
    /**
     * Create an OpenEMR API client
     * @param {Object} smartClient - The FHIR client from SMART on FHIR launch
     */
    constructor(smartClient) {
        this.smartClient = smartClient;
        this._accessToken = null;
        this._serverUrl = null;
    }

    /**
     * Get the access token from the SMART client
     * @returns {Promise<string>} The bearer token
     */
    async getAccessToken() {
        if (this._accessToken) {
            return this._accessToken;
        }

        // Get token from SMART client state
        const state = this.smartClient.state;
        if (state && state.tokenResponse && state.tokenResponse.access_token) {
            this._accessToken = state.tokenResponse.access_token;
            return this._accessToken;
        }

        throw new Error('No access token available from SMART client');
    }

    /**
     * Get the OpenEMR server base URL
     * @returns {Promise<string>} The server URL
     */
    async getServerUrl() {
        if (this._serverUrl) {
            return this._serverUrl;
        }

        const state = this.smartClient.state;
        if (state && state.serverUrl) {
            // FHIR URL is like https://localhost:9300/apis/default/fhir
            // We need https://localhost:9300/apis/default/api
            this._serverUrl = state.serverUrl.replace(/\/fhir\/?$/, '/api');
            return this._serverUrl;
        }

        throw new Error('No server URL available from SMART client');
    }

    /**
     * Make an authenticated API request
     * @param {string} method - HTTP method
     * @param {string} endpoint - API endpoint path
     * @param {Object} options - Fetch options
     * @returns {Promise<Object>} Response data
     */
    async request(method, endpoint, options = {}) {
        const token = await this.getAccessToken();
        const serverUrl = await this.getServerUrl();

        const url = `${serverUrl}${endpoint}`;
        const headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
            ...options.headers
        };

        const fetchOptions = {
            method,
            headers,
            ...options
        };

        if (options.body && typeof options.body === 'object') {
            fetchOptions.body = JSON.stringify(options.body);
        }

        console.log(`OpenEMR API: ${method} ${url}`);

        const response = await fetch(url, fetchOptions);

        if (!response.ok) {
            let errorMessage = `API request failed: ${response.status} ${response.statusText}`;
            try {
                const errorData = await response.json();
                if (errorData.error_description) {
                    errorMessage = errorData.error_description;
                } else if (errorData.error) {
                    errorMessage = errorData.error;
                }
            } catch (e) {
                // Ignore JSON parse errors
            }
            throw new Error(errorMessage);
        }

        if (response.status === 204 || response.headers.get('content-length') === '0') {
            return {};
        }

        return response.json();
    }

    // =========================================================================
    // Custom User Settings API
    // =========================================================================

    /**
     * Get all custom user settings for the current user
     * @returns {Promise<Object>} Settings data
     *
     * @example
     * const result = await api.getCustomUserSettings();
     * console.log(result.data); // Array of settings with values
     */
    async getCustomUserSettings() {
        return this.request('GET', '/user/settings/custom');
    }

    /**
     * Get all available custom field definitions (without values)
     * These are the USR layout fields configured in Admin > Layouts
     * @returns {Promise<Object>} Field definitions
     *
     * @example
     * const result = await api.getCustomUserSettingsFields();
     * console.log(result.data); // Array of field definitions
     */
    async getCustomUserSettingsFields() {
        return this.request('GET', '/user/settings/custom/fields');
    }

    /**
     * Get a specific custom user setting by field ID
     * @param {string} fieldId - The field identifier
     * @returns {Promise<Object>} Setting data
     *
     * @example
     * const result = await api.getCustomUserSetting('preferred_language');
     * console.log(result.data.field_value);
     */
    async getCustomUserSetting(fieldId) {
        return this.request('GET', `/user/settings/custom/${encodeURIComponent(fieldId)}`);
    }

    /**
     * Update a custom user setting value
     * @param {string} fieldId - The field identifier
     * @param {string} value - The new value to set
     * @returns {Promise<Object>} Update confirmation
     *
     * @example
     * await api.updateCustomUserSetting('preferred_language', 'es');
     */
    async updateCustomUserSetting(fieldId, value) {
        return this.request('PUT', `/user/settings/custom/${encodeURIComponent(fieldId)}`, {
            body: { field_value: value }
        });
    }

    /**
     * Delete (reset) a custom user setting to its default value
     * @param {string} fieldId - The field identifier
     * @returns {Promise<Object>} Deletion confirmation
     *
     * @example
     * await api.deleteCustomUserSetting('preferred_language');
     */
    async deleteCustomUserSetting(fieldId) {
        return this.request('DELETE', `/user/settings/custom/${encodeURIComponent(fieldId)}`);
    }

    // =========================================================================
    // Utility methods
    // =========================================================================

    /**
     * Get information about the current authenticated user
     * @returns {Promise<Object>} User data
     */
    async getCurrentUser() {
        return this.request('GET', '/user');
    }

    /**
     * Get a patient by UUID
     * @param {string} patientUuid - Patient UUID
     * @returns {Promise<Object>} Patient data
     */
    async getPatient(patientUuid) {
        return this.request('GET', `/patient/${encodeURIComponent(patientUuid)}`);
    }

    /**
     * Check if the API is accessible with current token
     * @returns {Promise<boolean>} True if API is accessible
     */
    async checkConnection() {
        try {
            await this.getCurrentUser();
            return true;
        } catch (error) {
            console.warn('OpenEMR API connection check failed:', error.message);
            return false;
        }
    }
}

// Export for use as module or global
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { OpenEMRApi };
} else {
    window.OpenEMRApi = OpenEMRApi;
}