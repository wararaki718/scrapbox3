import React, {useState} from 'react';

const Chat: React.FC = () => {
  const [client_id, setClientId] = useState(Date.now());
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState(new Array<string>());

  const ws = new WebSocket(`ws://localhost:5000/ws/${client_id}`);
  ws.onmessage = (event: any): void => {
    setMessages([...messages, event.data]);
  };

  const handleChange = (event: any): void => {
    setMessage(event.target.value);
  };

  const sendMessage = (event: any): void => {
      ws.send(message);
      setMessage("");
      event.preventDefault();
  };

  return (
    <div>
      <h2> Your ID: {client_id}</h2>
      <form action="" onSubmit={sendMessage}>
          <input type="text" autoComplete="off" value={message} onChange={handleChange} />
          <button>Send</button>
      </form>
      <ul>
        { messages.map(x => (<li key={Date.now()}>{x}</li>)) }
      </ul>
    </div>
  );
}

export default Chat;
