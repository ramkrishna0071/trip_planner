import { useState } from 'react';
import './PlanForm.css';

const defaultState = {
  origin: 'San Francisco',
  destinations: 'Paris, Amsterdam, Berlin',
  startDate: '2025-10-10',
  endDate: '2025-10-20',
  budget: '4500',
  adults: '2',
  children: '1',
  seniors: '0',
  objective: 'balanced',
};

function PlanForm({ onSubmit, loading }) {
  const [form, setForm] = useState(defaultState);

  const updateField = (key) => (event) => {
    setForm((prev) => ({ ...prev, [key]: event.target.value }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!onSubmit) {
      return;
    }

    const destinations = form.destinations
      .split(',')
      .map((entry) => entry.trim())
      .filter(Boolean);

    onSubmit({
      origin: form.origin.trim(),
      destinations,
      startDate: form.startDate,
      endDate: form.endDate,
      budget: Number.parseFloat(form.budget || '0'),
      adults: Number.parseInt(form.adults || '0', 10),
      children: Number.parseInt(form.children || '0', 10),
      seniors: Number.parseInt(form.seniors || '0', 10),
      objective: form.objective,
    });
  };

  return (
    <form className="plan-form" onSubmit={handleSubmit}>
      <div className="plan-form__header">
        <h2>Design a trip</h2>
        <p>Describe the travellers, target cities, dates, and overall budget.</p>
      </div>

      <div className="plan-form__grid">
        <label className="plan-form__field">
          <span>Origin</span>
          <input value={form.origin} onChange={updateField('origin')} placeholder="Home airport" />
        </label>

        <label className="plan-form__field">
          <span>Destinations</span>
          <input
            value={form.destinations}
            onChange={updateField('destinations')}
            placeholder="Comma-separated"
          />
        </label>

        <label className="plan-form__field">
          <span>Start date</span>
          <input type="date" value={form.startDate} onChange={updateField('startDate')} />
        </label>

        <label className="plan-form__field">
          <span>End date</span>
          <input type="date" value={form.endDate} onChange={updateField('endDate')} />
        </label>

        <label className="plan-form__field">
          <span>Budget (total)</span>
          <input type="number" min="0" value={form.budget} onChange={updateField('budget')} />
        </label>

        <label className="plan-form__field">
          <span>Objective</span>
          <select value={form.objective} onChange={updateField('objective')}>
            <option value="balanced">Balanced</option>
            <option value="cheapest">Cheapest</option>
            <option value="comfort">Comfort</option>
            <option value="family_friendly">Family friendly</option>
          </select>
        </label>

        <label className="plan-form__field">
          <span>Adults</span>
          <input type="number" min="0" value={form.adults} onChange={updateField('adults')} />
        </label>

        <label className="plan-form__field">
          <span>Children</span>
          <input type="number" min="0" value={form.children} onChange={updateField('children')} />
        </label>

        <label className="plan-form__field">
          <span>Seniors</span>
          <input type="number" min="0" value={form.seniors} onChange={updateField('seniors')} />
        </label>
      </div>

      <button type="submit" className="plan-form__submit" disabled={loading}>
        {loading ? 'Generating...' : 'Generate plan'}
      </button>
    </form>
  );
}

export default PlanForm;
