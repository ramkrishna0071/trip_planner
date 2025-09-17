import BookingLinks from './BookingLinks.jsx';
import './BundleCard.css';

function formatCurrency(value, currency) {
  try {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
      maximumFractionDigits: 0,
    }).format(value);
  } catch (error) {
    return `${currency} ${value.toFixed(0)}`;
  }
}

function BundleCard({ bundle }) {
  return (
    <article className="bundle-card">
      <header className="bundle-card__header">
        <div>
          <span className="bundle-card__label">{bundle.label}</span>
          <h3>{bundle.summary}</h3>
        </div>
        <div className="bundle-card__price">{formatCurrency(bundle.total_cost, bundle.currency)}</div>
      </header>

      <div className="bundle-card__section">
        <h4>Travel legs</h4>
        <ul>
          {bundle.travel.map((leg, index) => (
            <li key={`${leg.frm}-${leg.to}-${index}`}>
              <strong>{leg.mode}</strong> · {leg.frm} → {leg.to}
              {leg.date ? ` on ${leg.date}` : ''}
              {leg.duration_hr ? ` · ${leg.duration_hr.toFixed(1)}h` : ''}
              {typeof leg.cost_estimate === 'number'
                ? ` · ${formatCurrency(leg.cost_estimate, bundle.currency)}`
                : ''}
            </li>
          ))}
        </ul>
      </div>

      <div className="bundle-card__section bundle-card__section--grid">
        <div>
          <h4>Stays</h4>
          <ul>
            {bundle.stays.map((stay) => (
              <li key={`${stay.city}-${stay.style}`}>
                {stay.city}: {stay.nights} nights · {stay.style}
                {' · '}
                {formatCurrency(stay.budget_per_night, bundle.currency)} / night
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4>Daily highlights</h4>
          <ul className="bundle-card__experiences">
            {bundle.experience_plan.map((plan) => (
              <li key={plan.city}>
                <strong>{plan.city}</strong>
                <span>{plan.must_do[0]}</span>
                {plan.hidden_gem[0] ? <span className="bundle-card__gem">Hidden gem: {plan.hidden_gem[0]}</span> : null}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {bundle.booking_links?.length ? (
        <div className="bundle-card__section">
          <h4>Bundle booking links</h4>
          <BookingLinks links={bundle.booking_links} />
        </div>
      ) : null}

      {bundle.notes?.length ? (
        <div className="bundle-card__footer">
          <h4>Notes</h4>
          <ul>
            {bundle.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </article>
  );
}

export default BundleCard;
