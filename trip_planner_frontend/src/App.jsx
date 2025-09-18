import { useState } from 'react';
import PlanForm from './components/PlanForm.jsx';
import ChatPanel from './components/ChatPanel.jsx';
import PlanSummary from './components/PlanSummary.jsx';
import BundlesView from './components/BundlesView.jsx';
import SourceList from './components/SourceList.jsx';
import { sampleResponse } from './lib/sampleData.js';
import { buildRequestPayload } from './lib/payload.js';
import './App.css';

const rawApiBaseUrl =
  typeof import.meta.env.VITE_API_BASE_URL === 'string' && import.meta.env.VITE_API_BASE_URL.trim().length
    ? import.meta.env.VITE_API_BASE_URL
    : 'http://localhost:8000';

// Trim a trailing slash so request assembly is predictable regardless of how the
// base URL is configured (env var, default, etc.).
const API_BASE_URL = rawApiBaseUrl.replace(/\/$/, '');

function formatDestinations(destinations) {
  return destinations.length ? destinations.join(', ') : 'your selected cities';
}

const initialMessages = [
  {
    id: 'intro',
    role: 'assistant',
    content:
      'Share your origin, dream destinations, travel window, and budget. I will research travel legs, propose lodging mixes, and surface ready-to-book links.',
  },
];

function App() {
  const [messages, setMessages] = useState(initialMessages);
  const [planResult, setPlanResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [usedSampleFallback, setUsedSampleFallback] = useState(false);

  const bundles = planResult?.baseline_plan?.options ?? [];
  const sources = planResult?.agent_context?.sources ?? [];
  const llmSources = planResult?.agent_context?.llm_sources ?? [];
  const planNotes = planResult?.agent_context?.notes ?? [];

  const addMessage = (message) => {
    setMessages((prev) => [
      ...prev,
      { ...message, id: `${message.role}-${Date.now()}-${Math.random().toString(16).slice(2)}` },
    ]);
  };

  const handlePlanSubmit = async (formValues) => {
    const requestPayload = buildRequestPayload(formValues);

    addMessage({
      role: 'user',
      content: `I want to travel from ${requestPayload.origin} to ${formatDestinations(
        requestPayload.destinations
      )} between ${requestPayload.dates.start} and ${requestPayload.dates.end} with a budget of $${requestPayload.budget_total}.`,
    });

    setLoading(true);
    setError('');
    setUsedSampleFallback(false);

    try {
      if (!API_BASE_URL) {
        throw new Error('API base URL not configured.');
      }

      const response = await fetch(`${API_BASE_URL}/api/plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload),
      });

      if (!response.ok) {
        throw new Error(`Planner API responded with ${response.status}`);
      }

      const data = await response.json();
      setPlanResult(data);
      addMessage({
        role: 'assistant',
        content:
          data.trip_plan?.overview ??
          'Here is the latest itinerary including bundles, logistics, and booking shortcuts.',
      });
    } catch (fetchError) {
      console.warn('Falling back to sample response', fetchError);
      setPlanResult(sampleResponse);
      setError('Planner API unavailable. Showing interactive sample instead.');
      setUsedSampleFallback(true);
      addMessage({
        role: 'assistant',
        content:
          'Live scraping is offline, so I loaded a curated sample itinerary. Configure VITE_API_BASE_URL to connect your orchestrator.',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="app-shell__column app-shell__column--left">
        <ChatPanel messages={messages} />
        <PlanForm onSubmit={handlePlanSubmit} loading={loading} />
        {error ? <div className="app-shell__status app-shell__status--warning">{error}</div> : null}
        {usedSampleFallback ? (
          <div className="app-shell__status app-shell__status--info">
            Using offline sample data. Start the orchestrator API and set <code>VITE_API_BASE_URL</code> to see live web-sourced content.
          </div>
        ) : null}
      </div>
      <div className="app-shell__column app-shell__column--right">
        <PlanSummary tripPlan={planResult?.trip_plan} />
        {planNotes.length ? (
          <section className="app-shell__notes">
            <h3>Operational notes</h3>
            <ul>
              {planNotes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </section>
        ) : null}
        <BundlesView bundles={bundles} />
        <SourceList sources={sources} llmSources={llmSources} />
      </div>
    </div>
  );
}

export default App;
