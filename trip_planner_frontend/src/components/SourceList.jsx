import './SourceList.css';

function SourceList({ sources = [], llmSources = [] }) {
  if (!sources.length && !llmSources.length) {
    return null;
  }

  return (
    <section className="source-list">
      <h3>Sources</h3>
      <div className="source-list__grid">
        {sources.map((source) => (
          <article key={`${source.city}-${source.url}`}>
            <h4>{source.city}</h4>
            <a href={source.url} target="_blank" rel="noreferrer">
              {source.title}
            </a>
          </article>
        ))}
        {llmSources.map((entry) => (
          <article key={`llm-${entry.model}`} className="source-list__llm">
            <h4>Model</h4>
            <p>
              <strong>{entry.model}</strong>
            </p>
            {entry.reason ? <p>{entry.reason}</p> : null}
          </article>
        ))}
      </div>
    </section>
  );
}

export default SourceList;
