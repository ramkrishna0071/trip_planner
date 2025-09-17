import BundleCard from './BundleCard.jsx';
import './BundlesView.css';

function BundlesView({ bundles }) {
  if (!bundles?.length) {
    return (
      <section className="bundles-view bundles-view--empty">
        <p>Your tailored itineraries will appear here once a plan has been generated.</p>
      </section>
    );
  }

  return (
    <section className="bundles-view">
      {bundles.map((bundle) => (
        <BundleCard key={bundle.label} bundle={bundle} />
      ))}
    </section>
  );
}

export default BundlesView;
