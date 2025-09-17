import { sampleRequest } from './sampleData.js';

export function inferPurpose(objective) {
  switch (objective) {
    case 'family_friendly':
      return 'family vacation';
    case 'comfort':
      return 'premium leisure escape';
    case 'cheapest':
      return 'budget getaway';
    default:
      return 'leisure';
  }
}

export function buildRequestPayload(formValues = {}) {
  const destinations = Array.isArray(formValues.destinations)
    ? formValues.destinations.filter(Boolean)
    : [];

  return {
    ...sampleRequest,
    origin: formValues.origin || sampleRequest.origin,
    destinations: destinations.length ? destinations : sampleRequest.destinations,
    dates: {
      start: formValues.startDate || sampleRequest.dates.start,
      end: formValues.endDate || sampleRequest.dates.end,
    },
    budget_total:
      Number.isFinite(formValues.budget) && formValues.budget > 0
        ? formValues.budget
        : sampleRequest.budget_total,
    party: {
      adults: formValues.adults ?? sampleRequest.party.adults,
      children: formValues.children ?? sampleRequest.party.children,
      seniors: formValues.seniors ?? sampleRequest.party.seniors,
    },
    purpose: formValues.purpose?.trim() || inferPurpose(formValues.objective),
    prefs: {
      ...sampleRequest.prefs,
      objective: formValues.objective || sampleRequest.prefs.objective,
    },
  };
}
