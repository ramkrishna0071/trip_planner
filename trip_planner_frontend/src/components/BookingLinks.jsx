import './BookingLinks.css';

const CATEGORY_LABELS = {
  flight: 'Flights',
  train: 'Rail',
  bus: 'Coach',
  stay: 'Stays',
  local_pass: 'City passes',
  sightseeing: 'Experiences',
  transport: 'Transport',
};

function groupByCategory(links) {
  return links.reduce((acc, link) => {
    const bucket = acc.get(link.category) ?? [];
    bucket.push(link);
    acc.set(link.category, bucket);
    return acc;
  }, new Map());
}

function BookingLinks({ links }) {
  if (!links?.length) {
    return null;
  }

  const grouped = groupByCategory(links);

  return (
    <div className="booking-links">
      {[...grouped.entries()].map(([category, categoryLinks]) => (
        <div key={category} className="booking-links__group">
          <h4>{CATEGORY_LABELS[category] ?? category}</h4>
          <ul>
            {categoryLinks.map((link) => (
              <li key={`${category}-${link.label}`}>
                <a href={link.url} target="_blank" rel="noreferrer">
                  {link.label}
                </a>
                {link.details ? <span className="booking-links__details">{link.details}</span> : null}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

export default BookingLinks;
