/**
 * SMART FHIR Console Viewer
 * Fetch and display Conditions, Procedures, Observations, Medications, Allergies, and Encounters
 */
console.log("=== CONSOLEVIEWER.JS FILE LOADED ===");

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

        // Get patient ID
        const patientId = smartClient.patient.id;
        logToUI('Patient ID:', patientId);

        // ==================== FETCH CONDITIONS ====================
        let conditionData = null;
        let conditions = [];

        try {
            conditionData = await smartClient.patient.request("Condition");

            if (conditionData && conditionData.resourceType === 'Bundle') {
                conditions = conditionData.entry?.map(e => e.resource) || [];
            } else if (Array.isArray(conditionData)) {
                conditions = conditionData;
            }
        } catch (e) {
            logToUI('Error fetching conditions:', e.message);
        }

        // Parse conditions
        const { activeConditions, inactiveConditions } = parseConditions(conditions);

        // ==================== FETCH PROCEDURES ====================
        let procedureData = null;
        let procedures = [];

        try {
            procedureData = await smartClient.patient.request("Procedure");

            if (procedureData && procedureData.resourceType === 'Bundle') {
                procedures = procedureData.entry?.map(e => e.resource) || [];
            } else if (Array.isArray(procedureData)) {
                procedures = procedureData;
            }
        } catch (e) {
            logToUI('Error fetching procedures:', e.message);
        }

        // ==================== FETCH CURRENT ENCOUNTER ====================
        let encounterData = null;
        let currentEncounter = null;

        try {
            if (smartClient.encounter?.id) {
                currentEncounter = await smartClient.request(`Encounter/${smartClient.encounter.id}`);
            } else {
                encounterData = await smartClient.patient.request("Encounter?_sort=-date&_count=1");

                if (encounterData && encounterData.resourceType === 'Bundle' && encounterData.entry?.length > 0) {
                    currentEncounter = encounterData.entry[0].resource;
                }
            }
        } catch (e) {
            logToUI('Error fetching encounter:', e.message);
        }

        // ==================== FETCH ALL OBSERVATIONS FOR PATIENT ====================
        let allObservationData = null;
        let allObservations = [];

        try {
            allObservationData = await smartClient.patient.request("Observation");

            if (allObservationData && allObservationData.resourceType === 'Bundle') {
                allObservations = allObservationData.entry?.map(e => e.resource) || [];
            } else if (Array.isArray(allObservationData)) {
                allObservations = allObservationData;
            }
        } catch (e) {
            logToUI('Error fetching all observations:', e.message);
            allObservationData = { error: e.message };
        }

        // ==================== CATEGORIZE OBSERVATIONS FOR CURRENT ENCOUNTER ====================
        let vitals = [];
        let laboratoryTests = [];
        let imagingTests = [];
        let procedureTests = [];
        let examTests = [];
        let otherObservations = [];

        if (currentEncounter && allObservations.length > 0) {
            const encounterStart = currentEncounter.period?.start;

            if (encounterStart) {
                const encounterStartDate = new Date(encounterStart);

                const encounterObservations = allObservations.filter(obs => {
                    const obsDate = obs.effectiveDateTime || obs.issued;
                    if (!obsDate) return false;

                    const obsDateTime = new Date(obsDate);
                    return obsDateTime >= encounterStartDate;
                });

                for (const obs of encounterObservations) {
                    const category = obs.category?.[0]?.coding?.[0]?.code || '';

                    if (category === 'vital-signs') {
                        vitals.push(obs);
                    } else if (category === 'social-history') {
                        continue;
                    } else if (category === 'laboratory') {
                        laboratoryTests.push(obs);
                    } else if (category === 'imaging') {
                        imagingTests.push(obs);
                    } else if (category === 'procedure') {
                        procedureTests.push(obs);
                    } else if (category === 'exam') {
                        examTests.push(obs);
                    } else {
                        otherObservations.push(obs);
                    }
                }
            }
        }

        // ==================== FETCH MEDICATIONS ====================
        let medicationData = null;
        let medications = [];

        try {
            medicationData = await smartClient.patient.request("MedicationRequest");

            if (medicationData && medicationData.resourceType === 'Bundle') {
                medications = medicationData.entry?.map(e => e.resource) || [];
            } else if (Array.isArray(medicationData)) {
                medications = medicationData;
            }
        } catch (e) {
            logToUI('Error fetching medications:', e.message);
            medicationData = { error: e.message };
        }

        const { activeMeds, inactiveMeds } = parseMedications(medications);

        // ==================== FETCH ALLERGIES ====================
        let allergyData = null;
        let allergies = [];

        try {
            allergyData = await smartClient.patient.request("AllergyIntolerance");

            if (allergyData && allergyData.resourceType === 'Bundle') {
                allergies = allergyData.entry?.map(e => e.resource) || [];
            } else if (Array.isArray(allergyData)) {
                allergies = allergyData;
            }
        } catch (e) {
            logToUI('Error fetching allergies:', e.message);
            allergyData = { error: e.message };
        }

        const { activeAllergies, inactiveAllergies } = parseAllergies(allergies);

        // ==================== DISPLAY FORMATTED OUTPUT ====================
        logToUI('');
        logToUI('========================================');
        logToUI('CURRENT ENCOUNTER');
        logToUI('========================================');
        if (currentEncounter) {
            logToUI('Encounter ID:', currentEncounter.id);
            logToUI('Date:', currentEncounter.period?.start || currentEncounter.meta?.lastUpdated || 'Unknown');
            logToUI('Type:', currentEncounter.type?.[0]?.coding?.[0]?.display ||
                currentEncounter.type?.[0]?.text ||
                currentEncounter.class?.display ||
                currentEncounter.class?.code ||
                'Unknown');
            logToUI('Status:', currentEncounter.status || 'Unknown');
        } else {
            logToUI('No current encounter found');
        }

        // ==================== VITALS FOR CURRENT ENCOUNTER ====================
        logToUI('');
        logToUI('========================================');
        logToUI('VITAL SIGNS (Current Encounter)');
        logToUI('========================================');
        if (vitals.length > 0) {
            const groupedVitals = groupObservationsByPanel(vitals);
            displayGroupedObservations(groupedVitals);
        } else {
            logToUI('No vital signs recorded for current encounter');
        }

        // ==================== OTHER OBSERVATIONS FOR CURRENT ENCOUNTER ====================
        logToUI('');
        logToUI('========================================');
        logToUI('OTHER OBSERVATIONS (Current Encounter)');
        logToUI('========================================');

        // Combine all non-vital observations for current encounter
        const allOtherCurrentEncounter = [
            ...laboratoryTests,
            ...imagingTests,
            ...procedureTests,
            ...examTests,
            ...otherObservations
        ];

        if (allOtherCurrentEncounter.length > 0) {
            const groupedOther = groupObservationsByPanel(allOtherCurrentEncounter);
            displayGroupedObservations(groupedOther);
        } else {
            logToUI('No other observations recorded for current encounter');
        }

        // ==================== PAST MEDICAL HISTORY ====================
        logToUI('');
        logToUI('========================================');
        logToUI('PAST MEDICAL HISTORY');
        logToUI('========================================');
        logToUI('');
        logToUI('Active Conditions:');
        if (activeConditions.length > 0) {
            for (const condition of activeConditions) {
                logToUI('  - ' + condition);
            }
        } else {
            logToUI('  None');
        }
        logToUI('');
        logToUI('Inactive Conditions:');
        if (inactiveConditions.length > 0) {
            for (const condition of inactiveConditions) {
                logToUI('  - ' + condition);
            }
        } else {
            logToUI('  None');
        }

        // ==================== MEDICATIONS ====================
        logToUI('');
        logToUI('========================================');
        logToUI('MEDICATIONS');
        logToUI('========================================');
        logToUI('');
        logToUI('Active Medications:');
        if (activeMeds.length > 0) {
            for (const med of activeMeds) {
                logToUI('  - ' + med);
            }
        } else {
            logToUI('  None');
        }
        logToUI('');
        logToUI('Inactive Medications:');
        if (inactiveMeds.length > 0) {
            for (const med of inactiveMeds) {
                logToUI('  - ' + med);
            }
        } else {
            logToUI('  None');
        }

        // ==================== ALLERGIES ====================
        logToUI('');
        logToUI('========================================');
        logToUI('ALLERGIES');
        logToUI('========================================');
        logToUI('');
        logToUI('Active Allergies:');
        if (activeAllergies.length > 0) {
            for (const allergy of activeAllergies) {
                logToUI('  - ' + allergy);
            }
        } else {
            logToUI('  None');
        }
        logToUI('');
        logToUI('Inactive Allergies:');
        if (inactiveAllergies.length > 0) {
            for (const allergy of inactiveAllergies) {
                logToUI('  - ' + allergy);
            }
        } else {
            logToUI('  None');
        }

        // ==================== PAST OBSERVATIONS (Historical - Excluding Vitals) ====================
        logToUI('');
        logToUI('========================================');
        logToUI('PAST OBSERVATIONS');
        logToUI('========================================');

        // Filter out vital-signs from historical observations
        const nonVitalObservations = allObservations.filter(obs => {
            const category = obs.category?.[0]?.coding?.[0]?.code || '';
            return category !== 'vital-signs';
        });

        if (nonVitalObservations.length > 0) {
            const groupedAll = groupObservationsByPanel(nonVitalObservations);
            displayGroupedObservations(groupedAll);
        } else {
            logToUI('No observations found for this patient');
        }

    } catch (err) {
        logToUI('An error occurred:', err.stack || err.message);
    }
}

/**
 * Group observations by their panel/test name
 * Panels have hasMember references, individual tests don't
 */
function groupObservationsByPanel(observations) {
    const groups = {};
    const standaloneTests = [];
    const processedIds = new Set();

    // First, identify panels (observations with hasMember)
    const panels = observations.filter(obs => obs.hasMember && obs.hasMember.length > 0);

    for (const panel of panels) {
        const panelName = panel.code?.text ||
            panel.code?.coding?.[0]?.display ||
            'Unknown Panel';

        const panelDate = panel.effectiveDateTime || panel.issued || '';

        // Get member observation IDs
        const memberIds = panel.hasMember.map(member => {
            const ref = member.reference || '';
            return ref.replace('Observation/', '');
        });

        // Find member observations
        const members = observations.filter(obs => memberIds.includes(obs.id));

        if (members.length > 0) {
            const groupKey = `${panelName}|||${panelDate}`;
            groups[groupKey] = {
                name: panelName,
                date: panelDate,
                tests: []
            };

            for (const member of members) {
                processedIds.add(member.id);
                const testInfo = extractTestValue(member);
                if (testInfo) {
                    groups[groupKey].tests.push(testInfo);
                }
            }
        }

        processedIds.add(panel.id);
    }

    // Process remaining observations (not part of any panel)
    for (const obs of observations) {
        if (processedIds.has(obs.id)) continue;
        if (obs.hasMember && obs.hasMember.length > 0) continue; // Skip panels without members
        if (obs.dataAbsentReason) continue; // Skip absent values

        const testInfo = extractTestValue(obs);
        if (testInfo) {
            standaloneTests.push(testInfo);
        }
    }

    // Group standalone tests by name (for repeated measurements like vitals)
    const standaloneGroups = {};
    for (const test of standaloneTests) {
        const key = test.name;
        if (!standaloneGroups[key]) {
            standaloneGroups[key] = {
                name: test.name,
                date: test.date,
                tests: []
            };
        }
        standaloneGroups[key].tests.push(test);
    }

    // Merge panel groups and standalone groups
    return { ...groups, ...standaloneGroups };
}

/**
 * Extract test name and value from an observation
 */
function extractTestValue(obs) {
    if (obs.dataAbsentReason) return null;

    const testName = obs.code?.text ||
        obs.code?.coding?.[0]?.display ||
        'Unknown Test';

    let value = '';

    // Check for component values (like blood pressure)
    if (obs.component && obs.component.length > 0) {
        const componentValues = [];
        for (const component of obs.component) {
            if (component.dataAbsentReason) continue;

            const componentName = component.code?.text ||
                component.code?.coding?.[0]?.display ||
                'Unknown';

            if (component.valueQuantity) {
                const unit = component.valueQuantity.unit || '';
                componentValues.push(`${componentName}: ${component.valueQuantity.value} ${unit}`.trim());
            }
        }

        if (componentValues.length > 0) {
            value = componentValues.join(', ');
        }
    }
    // Check for direct value
    else if (obs.valueQuantity) {
        const unit = obs.valueQuantity.unit || '';
        value = `${obs.valueQuantity.value} ${unit}`.trim();
    } else if (obs.valueString) {
        value = obs.valueString;
    } else if (obs.valueCodeableConcept) {
        value = obs.valueCodeableConcept.text ||
            obs.valueCodeableConcept.coding?.[0]?.display ||
            '';
    }

    if (!value) return null;

    const date = obs.effectiveDateTime || obs.issued || '';

    return {
        name: testName,
        value: value,
        date: date
    };
}

/**
 * Display grouped observations in a clean format
 */
function displayGroupedObservations(groups) {
    const sortedKeys = Object.keys(groups).sort();

    for (const key of sortedKeys) {
        const group = groups[key];
        logToUI('');
        logToUI(group.name + (group.date ? ` (${formatDate(group.date)})` : '') + ':');

        for (const test of group.tests) {
            if (group.tests.length === 1 && test.name === group.name) {
                // Single test with same name as group - just show value
                logToUI('  ' + test.value);
            } else {
                // Multiple tests or different name - show name: value
                logToUI('  ' + test.name + ': ' + test.value);
            }
        }
    }
}

/**
 * Format ISO date to readable format
 */
function formatDate(isoDate) {
    if (!isoDate) return '';
    try {
        const date = new Date(isoDate);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
        return isoDate;
    }
}

function parseConditions(conditions) {
    const activeConditions = [];
    const inactiveConditions = [];

    for (const cond of conditions) {
        const status = cond.clinicalStatus?.coding?.[0]?.code || 'unknown';

        const conditionName = cond.code?.text ||
            cond.code?.coding?.[0]?.display ||
            'Unknown condition';

        if (status === 'active') {
            activeConditions.push(conditionName);
        } else {
            inactiveConditions.push(conditionName);
        }
    }

    return { activeConditions, inactiveConditions };
}

function parseMedications(medications) {
    const activeMeds = [];
    const inactiveMeds = [];

    for (const med of medications) {
        const status = med.status || 'unknown';

        const medicationConcept = med.medicationCodeableConcept;
        let medName = medicationConcept?.coding?.[0]?.display ||
            medicationConcept?.text ||
            'Unknown medication';

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
        const status = allergy.clinicalStatus?.coding?.[0]?.code || 'unknown';

        const allergyCode = allergy.code;
        let allergyName = allergyCode?.coding?.[0]?.display ||
            allergyCode?.text ||
            'Unknown allergy';

        const category = allergy.category?.[0];
        if (category) {
            allergyName += ` [${category}]`;
        }

        const criticality = allergy.criticality;
        if (criticality) {
            allergyName += ` (Criticality: ${criticality})`;
        }

        const reaction = allergy.reaction?.[0]?.manifestation?.[0]?.coding?.[0]?.display ||
            allergy.reaction?.[0]?.manifestation?.[0]?.text;
        if (reaction) {
            allergyName += ` - Reaction: ${reaction}`;
        }

        const severity = allergy.reaction?.[0]?.severity;
        if (severity) {
            allergyName += ` (${severity})`;
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
