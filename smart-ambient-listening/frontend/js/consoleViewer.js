/**
 * SMART FHIR Console Viewer
 * Fetch and display Conditions, Procedures, Observations, Medications, and Allergies with SNOMED lookup
 */
console.log("=== CONSOLEVIEWER.JS FILE LOADED ===");

// SNOMED International Browser API (CORS enabled)
const SNOMED_API_URL = 'https://browser.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN/concepts';

function logToUI(...args) {
    const outputDiv = document.getElementById('console-output');
    if (!outputDiv) {
        console.log('[consoleViewer - no UI]', ...args);
        return;
    }

    const text = args
        .map(a => (typeof a === 'object' ? JSON.stringify(a, null, 2) : a))
        .join(' ') + '\n';

    outputDiv.textContent += text;
    console.log('[consoleViewer]', ...args);
}

async function lookupSnomedCode(code) {
    try {
        const response = await fetch(`${SNOMED_API_URL}/${code}`, {
            headers: {
                'Accept': 'application/json',
                'Accept-Language': 'en'
            }
        });

        if (!response.ok) {
            return null;
        }

        const data = await response.json();
        // pt = preferred term, fsn = fully specified name
        return data.pt?.term || data.fsn?.term || null;
    } catch (e) {
        console.log('SNOMED lookup error for code', code, ':', e.message);
        return null;
    }
}

async function initConsoleViewer() {
    logToUI('Console viewer initializing...');

    try {
        // Wait for app.js to populate the patient banner (confirms SMART context is ready)
        let patientBannerReady = false;
        let attempts = 0;

        while (!patientBannerReady && attempts < 20) {
            const nameElement = document.getElementById('patient-name');
            patientBannerReady = nameElement?.textContent?.trim() ? true : false;

            if (!patientBannerReady) {
                logToUI('Waiting for app.js to load... attempt', attempts + 1);
                await new Promise(r => setTimeout(r, 500));
                attempts++;
            }
        }

        if (!patientBannerReady) {
            logToUI('Error: App not ready after waiting.');
            return;
        }

        logToUI('App ready. Fetching data...');

        // Get SMART client
        const smartClient = await FHIR.oauth2.ready();

        // ==================== FETCH CONDITIONS ====================
        logToUI('');
        logToUI('Fetching conditions...');
        let conditionData = null;
        let conditions = [];

        try {
            conditionData = await smartClient.patient.request("Condition");

            if (conditionData && conditionData.resourceType === 'Bundle') {
                conditions = conditionData.entry?.map(e => e.resource) || [];
            } else if (Array.isArray(conditionData)) {
                conditions = conditionData;
            }

            logToUI('Conditions fetched:', conditions.length);
        } catch (e) {
            logToUI('Error fetching conditions:', e.message);
        }

        // Test SNOMED lookup if we have conditions
        let activeConditions = [];
        let inactiveConditions = [];

        if (conditions.length > 0) {
            logToUI('');
            logToUI('Testing SNOMED International API lookup...');
            const firstCode = conditions[0]?.code?.coding?.[0]?.code;
            if (firstCode) {
                logToUI('Testing lookup for code:', firstCode);
                const testResult = await lookupSnomedCode(firstCode);
                if (testResult) {
                    logToUI('Lookup SUCCESS! Display name:', testResult);
                } else {
                    logToUI('Lookup failed for test code');
                }
            }

            // Parse conditions with SNOMED lookup
            logToUI('');
            logToUI('Looking up all SNOMED codes (this may take a moment)...');
            const parsedConditions = await parseConditions(conditions);
            activeConditions = parsedConditions.active;
            inactiveConditions = parsedConditions.inactive;
        }

        // ==================== FETCH PROCEDURES ====================
        logToUI('');
        logToUI('');
        logToUI('Fetching procedures...');
        let procedureData = null;
        let procedures = [];

        try {
            procedureData = await smartClient.patient.request("Procedure");

            if (procedureData && procedureData.resourceType === 'Bundle') {
                procedures = procedureData.entry?.map(e => e.resource) || [];
            } else if (Array.isArray(procedureData)) {
                procedures = procedureData;
            }

            logToUI('Procedures fetched:', procedures.length);
        } catch (e) {
            logToUI('Error fetching procedures:', e.message);
        }

        // ==================== FETCH OBSERVATIONS (by category) ====================
        logToUI('');
        logToUI('');
        logToUI('Fetching observations...');
        let observationData = null;
        let observations = [];

        // Try laboratory observations first
        try {
            logToUI('Trying laboratory observations...');
            observationData = await smartClient.patient.request("Observation?category=laboratory");

            if (observationData && observationData.resourceType === 'Bundle') {
                observations = observationData.entry?.map(e => e.resource) || [];
            } else if (Array.isArray(observationData)) {
                observations = observationData;
            }

            logToUI('Laboratory observations fetched:', observations.length);
        } catch (e) {
            logToUI('Error fetching laboratory observations:', e.message);
        }

        // Try vital-signs observations
        try {
            logToUI('Trying vital-signs observations...');
            const vitalData = await smartClient.patient.request("Observation?category=vital-signs");

            let vitals = [];
            if (vitalData && vitalData.resourceType === 'Bundle') {
                vitals = vitalData.entry?.map(e => e.resource) || [];
            } else if (Array.isArray(vitalData)) {
                vitals = vitalData;
            }

            logToUI('Vital-signs observations fetched:', vitals.length);

            // Combine with laboratory observations
            observations = [...observations, ...vitals];

            // Combine raw data for display
            if (observationData && vitalData) {
                observationData = {
                    laboratory: observationData,
                    vitalSigns: vitalData
                };
            } else if (vitalData) {
                observationData = vitalData;
            }
        } catch (e) {
            logToUI('Error fetching vital-signs observations:', e.message);
        }

        logToUI('Total observations fetched:', observations.length);

        // ==================== FETCH MEDICATIONS ====================
        logToUI('');
        logToUI('');
        logToUI('Fetching medications...');
        let medicationData = null;
        let medications = [];

        try {
            medicationData = await smartClient.patient.request("MedicationRequest");

            if (medicationData && medicationData.resourceType === 'Bundle') {
                medications = medicationData.entry?.map(e => e.resource) || [];
            } else if (Array.isArray(medicationData)) {
                medications = medicationData;
            }

            logToUI('Medications fetched:', medications.length);
        } catch (e) {
            logToUI('Error fetching medications:', e.message);
            logToUI('Note: This may be a server issue or the patient has no medications.');
            medicationData = { error: e.message, note: 'Server returned error - possibly no medications for this patient' };
        }

        // Parse medications
        const { activeMeds, inactiveMeds } = parseMedications(medications);

        // ==================== FETCH ALLERGIES ====================
        logToUI('');
        logToUI('');
        logToUI('Fetching allergies...');
        let allergyData = null;
        let allergies = [];

        try {
            allergyData = await smartClient.patient.request("AllergyIntolerance");

            if (allergyData && allergyData.resourceType === 'Bundle') {
                allergies = allergyData.entry?.map(e => e.resource) || [];
            } else if (Array.isArray(allergyData)) {
                allergies = allergyData;
            }

            logToUI('Allergies fetched:', allergies.length);
        } catch (e) {
            logToUI('Error fetching allergies:', e.message);
            logToUI('Note: This may be a server issue or the patient has no allergies.');
            allergyData = { error: e.message, note: 'Server returned error - possibly no allergies for this patient' };
        }

        // Parse allergies
        const { activeAllergies, inactiveAllergies } = parseAllergies(allergies);

        // ==================== DISPLAY FORMATTED OUTPUT ====================
        logToUI('');
        logToUI('');
        logToUI('========================================');
        logToUI('PAST MEDICAL HISTORY');
        logToUI('========================================');
        logToUI('');
        logToUI('Active Conditions: ' + (activeConditions.length > 0 ? activeConditions.join(', ') : 'None'));
        logToUI('');
        logToUI('Inactive Conditions: ' + (inactiveConditions.length > 0 ? inactiveConditions.join(', ') : 'None'));
        logToUI('');
        logToUI('========================================');
        logToUI('Total Conditions: ' + conditions.length + ' (Active: ' + activeConditions.length + ', Inactive: ' + inactiveConditions.length + ')');

        logToUI('');
        logToUI('');
        logToUI('========================================');
        logToUI('MEDICATIONS');
        logToUI('========================================');
        logToUI('');
        logToUI('Active Medications: ' + (activeMeds.length > 0 ? activeMeds.join(', ') : 'None'));
        logToUI('');
        logToUI('Inactive Medications: ' + (inactiveMeds.length > 0 ? inactiveMeds.join(', ') : 'None'));
        logToUI('');
        logToUI('========================================');
        logToUI('Total Medications: ' + medications.length + ' (Active: ' + activeMeds.length + ', Inactive: ' + inactiveMeds.length + ')');

        logToUI('');
        logToUI('');
        logToUI('========================================');
        logToUI('ALLERGIES');
        logToUI('========================================');
        logToUI('');
        logToUI('Active Allergies: ' + (activeAllergies.length > 0 ? activeAllergies.join(', ') : 'None'));
        logToUI('');
        logToUI('Inactive Allergies: ' + (inactiveAllergies.length > 0 ? inactiveAllergies.join(', ') : 'None'));
        logToUI('');
        logToUI('========================================');
        logToUI('Total Allergies: ' + allergies.length + ' (Active: ' + activeAllergies.length + ', Inactive: ' + inactiveAllergies.length + ')');

        // ==================== RAW DATA OUTPUT ====================
        logToUI('');
        logToUI('');
        logToUI('========================================');
        logToUI('RAW CONDITION DATA FROM FHIR SERVER');
        logToUI('(Note: "display" field is empty - resolved via SNOMED International API)');
        logToUI('========================================');
        logToUI('');
        logToUI(conditionData);

        logToUI('');
        logToUI('');
        logToUI('========================================');
        logToUI('RAW PROCEDURE DATA FROM FHIR SERVER');
        logToUI('========================================');
        logToUI('');
        logToUI(procedureData);

        logToUI('');
        logToUI('');
        logToUI('========================================');
        logToUI('RAW OBSERVATION DATA FROM FHIR SERVER');
        logToUI('========================================');
        logToUI('');
        logToUI(observationData);

        logToUI('');
        logToUI('');
        logToUI('========================================');
        logToUI('RAW MEDICATION DATA FROM FHIR SERVER');
        logToUI('========================================');
        logToUI('');
        logToUI(medicationData);

        logToUI('');
        logToUI('');
        logToUI('========================================');
        logToUI('RAW ALLERGY DATA FROM FHIR SERVER');
        logToUI('========================================');
        logToUI('');
        logToUI(allergyData);

    } catch (err) {
        logToUI('An error occurred:', err.stack || err.message);
    }
}

async function parseConditions(conditions) {
    const active = [];
    const inactive = [];

    // Get unique SNOMED codes first
    const uniqueCodes = [...new Set(
        conditions
            .map(c => c.code?.coding?.[0]?.code)
            .filter(Boolean)
    )];

    logToUI('Unique SNOMED codes to lookup:', uniqueCodes.length);

    // Lookup all codes and cache results
    const codeDisplayMap = {};
    let successCount = 0;

    for (const code of uniqueCodes) {
        const displayName = await lookupSnomedCode(code);
        if (displayName) {
            codeDisplayMap[code] = displayName;
            successCount++;
        } else {
            codeDisplayMap[code] = `SNOMED: ${code}`;
        }
        // Small delay to avoid rate limiting
        await new Promise(r => setTimeout(r, 150));
    }

    logToUI('Lookup complete. Resolved', successCount, 'of', uniqueCodes.length, 'codes.');

    // Now parse conditions using the cached lookups
    for (const cond of conditions) {
        const status = cond.clinicalStatus?.coding?.[0]?.code || 'unknown';
        const coding = cond.code?.coding?.[0];
        const code = coding?.code;

        // Use display if available, otherwise use lookup result
        const conditionName = coding?.display || codeDisplayMap[code] || `SNOMED: ${code}`;

        if (status === 'active') {
            active.push(conditionName);
        } else {
            inactive.push(conditionName);
        }
    }

    return { active, inactive };
}

function parseMedications(medications) {
    const activeMeds = [];
    const inactiveMeds = [];

    for (const med of medications) {
        // Get medication status
        const status = med.status || 'unknown';

        // Get medication name - try display from coding first, then text
        const medicationConcept = med.medicationCodeableConcept;
        let medName = medicationConcept?.coding?.[0]?.display ||
            medicationConcept?.text ||
            'Unknown medication';

        // Add dosage instruction if available
        const dosage = med.dosageInstruction?.[0]?.text;
        if (dosage) {
            medName += ` (${dosage.trim()})`;
        }

        if (status === 'active') {
            activeMeds.push(medName);
        } else {
            inactiveMeds.push(medName);
        }
    }

    return { activeMeds, inactiveMeds };
}

function parseAllergies(allergies) {
    const activeAllergies = [];
    const inactiveAllergies = [];

    for (const allergy of allergies) {
        // Get clinical status (active, inactive, resolved)
        const status = allergy.clinicalStatus?.coding?.[0]?.code || 'unknown';

        // Get allergy name - try code display first, then text
        const allergyCode = allergy.code;
        let allergyName = allergyCode?.coding?.[0]?.display ||
            allergyCode?.text ||
            'Unknown allergy';

        // Add reaction if available
        const reaction = allergy.reaction?.[0]?.manifestation?.[0]?.coding?.[0]?.display ||
            allergy.reaction?.[0]?.manifestation?.[0]?.text;
        if (reaction) {
            allergyName += ` (Reaction: ${reaction})`;
        }

        // Add severity if available
        const severity = allergy.reaction?.[0]?.severity;
        if (severity) {
            allergyName += ` [${severity}]`;
        }

        if (status === 'active') {
            activeAllergies.push(allergyName);
        } else {
            inactiveAllergies.push(allergyName);
        }
    }

    return { activeAllergies, inactiveAllergies };
}

// Wait for DOM to be ready before initializing
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initConsoleViewer);
} else {
    initConsoleViewer();
}