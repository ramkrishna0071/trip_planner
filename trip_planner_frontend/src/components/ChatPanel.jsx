import './ChatPanel.css';

function ChatBubble({ role, content, timestamp }) {
  const isUser = role === 'user';
  return (
    <div className={`chat-bubble ${isUser ? 'chat-bubble--user' : 'chat-bubble--assistant'}`}>
      <div className="chat-bubble__meta">
        <span className="chat-bubble__role">{isUser ? 'You' : 'Trip Studio'}</span>
        {timestamp ? <span className="chat-bubble__time">{timestamp}</span> : null}
      </div>
      <p>{content}</p>
    </div>
  );
}

function ChatPanel({ messages }) {
  return (
    <section className="chat-panel">
      <header className="chat-panel__header">
        <div>
          <h1>Trip planning workspace</h1>
          <p>Inspired by conversational planners like Mindtrip and Airial.</p>
        </div>
        <span className="chat-panel__badge">Preview</span>
      </header>
      <div className="chat-panel__conversation">
        {messages.map((message) => (
          <ChatBubble key={message.id} {...message} />
        ))}
      </div>
    </section>
  );
}

export default ChatPanel;
