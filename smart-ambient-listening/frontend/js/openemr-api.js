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
     * Get the OpenEMR REST API base URL
     * @returns {Promise<string>} The REST API URL
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
     * Get the FHIR base URL (for FHIR-scoped requests)
     * @returns {Promise<string>} The FHIR base URL
     */
    async getFhirUrl() {
        const state = this.smartClient.state;
        if (state && state.serverUrl) {
            return state.serverUrl.replace(/\/?$/, '');  // ensure no trailing slash
        }
        throw new Error('No FHIR server URL available from SMART client');
    }

    /**
     * Make an authenticated request to the FHIR API
     * (Uses the FHIR base URL instead of the REST API URL)
     * @param {string} method - HTTP method
     * @param {string} endpoint - FHIR endpoint path (e.g., /Encounter?patient=...)
     * @param {Object} options - Fetch options
     * @returns {Promise<Object>} Response data
     */
    async fhirRequest(method, endpoint, options = {}) {
        const token = await this.getAccessToken();
        const fhirUrl = await this.getFhirUrl();

        const url = `${fhirUrl}${endpoint}`;
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

        console.log(`OpenEMR FHIR: ${method} ${url}`);

        const response = await fetch(url, fetchOptions);

        if (!response.ok) {
            let errorMessage = `FHIR request failed: ${response.status} ${response.statusText}`;
            try {
                const errorData = await response.json();
                if (errorData.issue && errorData.issue[0]) {
                    errorMessage = errorData.issue[0].diagnostics || errorData.issue[0].details?.text || errorMessage;
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
     */
    async getCustomUserSettings() {
        return this.request('GET', '/user/settings/custom');
    }

    /**
     * Get all available custom field definitions (without values)
     * @returns {Promise<Object>} Field definitions
     */
    async getCustomUserSettingsFields() {
        return this.request('GET', '/user/settings/custom/fields');
    }

    /**
     * Get a specific custom user setting by field ID
     * @param {string} fieldId - The field identifier
     * @returns {Promise<Object>} Setting data
     */
    async getCustomUserSetting(fieldId) {
        return this.request('GET', `/user/settings/custom/${encodeURIComponent(fieldId)}`);
    }

    /**
     * Update a custom user setting value
     * @param {string} fieldId - The field identifier
     * @param {string} value - The new value to set
     * @returns {Promise<Object>} Update confirmation
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
     */
    async deleteCustomUserSetting(fieldId) {
        return this.request('DELETE', `/user/settings/custom/${encodeURIComponent(fieldId)}`);
    }

    // =========================================================================
    // Encounter API
    // =========================================================================

    /**
     * Get encounters for a specific patient via FHIR API
     * Uses patient/Encounter.read scope (already granted in SMART launch)
     * @param {string} patientId - Patient FHIR ID (uuid)
     * @returns {Promise<Array>} Array of encounter objects
     *
     * @example
     * const encounters = await api.getPatientEncounters('patient-uuid');
     * // Returns array of { id, date, reason, ... }
     */
    async getPatientEncounters(patientId) {
        const bundle = await this.fhirRequest('GET', `/Encounter?patient=${encodeURIComponent(patientId)}&_sort=-date&_count=20`);

        // FHIR returns a Bundle; extract entries
        if (!bundle.entry || bundle.entry.length === 0) {
            return [];
        }

        return bundle.entry.map(entry => {
            const enc = entry.resource;
            return {
                uuid: enc.id,
                id: enc.id,
                date: enc.period?.start || enc.meta?.lastUpdated || '',
                reason: enc.reasonCode?.[0]?.text || enc.type?.[0]?.text || '',
                status: enc.status || '',
                facility: enc.serviceProvider?.display || '',
                // Keep the full resource for reference
                _resource: enc
            };
        });
    }

    /**
     * Save a SOAP note to a specific encounter via the summarization backend.
     *
     * Routes through the backend service which has server-side access to OpenEMR,
     * avoiding SMART scope limitations on the REST API.
     *
     * @param {string} patientUuid - Patient UUID
     * @param {string} encounterUuid - Encounter UUID (FHIR ID)
     * @param {Object} soapData - SOAP note fields
     * @param {string} soapData.subjective - Subjective section text
     * @param {string} soapData.objective - Objective section text
     * @param {string} soapData.assessment - Assessment section text
     * @param {string} soapData.plan - Plan section text
     * @returns {Promise<Object>} Save confirmation
     */
    async saveSoapNote(patientUuid, encounterUuid, soapData) {
        const token = await this.getAccessToken();

        // Use the summarization service URL from SMART_CONFIG to route through backend
        const baseUrl = window.SMART_CONFIG?.summarizationServiceUrl || (window.location.origin + '/api/summarize');

        const url = `${baseUrl}/save-soap-note`;
        console.log(`OpenEMR Save SOAP: POST ${url}`);

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                patient_uuid: patientUuid,
                encounter_uuid: encounterUuid,
                subjective: soapData.subjective || '',
                objective: soapData.objective || '',
                assessment: soapData.assessment || '',
                plan: soapData.plan || ''
            })
        });

        if (!response.ok) {
            let errorMessage = `Save failed: ${response.status} ${response.statusText}`;
            try {
                const errorData = await response.json();
                if (errorData.detail) {
                    errorMessage = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
                } else if (errorData.error) {
                    errorMessage = errorData.error;
                }
            } catch (e) {
                // Ignore
            }
            throw new Error(errorMessage);
        }

        return response.json();
    }

    /**
     * Try to get the current encounter from SMART launch context.
     * Returns null if no encounter context is available.
     * @returns {Promise<string|null>} Encounter ID or null
     */
    async getEncounterFromContext() {
        try {
            const state = this.smartClient.state;

            // Check if encounter was part of the SMART launch context
            if (state && state.tokenResponse && state.tokenResponse.encounter) {
                return state.tokenResponse.encounter;
            }

            // Also check the launch context parameter
            if (state && state.encounter) {
                return state.encounter;
            }

            return null;
        } catch (error) {
            console.warn('Could not get encounter from SMART context:', error);
            return null;
        }
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