import BookingLinks from './BookingLinks.jsx';
import './PlanSummary.css';

function PlanSummary({ tripPlan }) {
  if (!tripPlan) {
    return (
      <section className="plan-summary plan-summary--empty">
        <h3>Ready when you are</h3>
        <p>Generate a plan to unlock booking links, timelines, and curated experiences.</p>
      </section>
    );
  }

  return (
    <section className="plan-summary">
      <header>
        <h2>{tripPlan.title}</h2>
        <p>{tripPlan.overview}</p>
      </header>

      {tripPlan.summary_points?.length ? (
        <ul className="plan-summary__bullets">
          {tripPlan.summary_points.map((point) => (
            <li key={point}>{point}</li>
          ))}
        </ul>
      ) : null}

      <div className="plan-summary__links">
        <h3>Booking shortcuts</h3>
        <BookingLinks links={tripPlan.booking_links} />
      </div>
    </section>
  );
}

export default PlanSummary;
