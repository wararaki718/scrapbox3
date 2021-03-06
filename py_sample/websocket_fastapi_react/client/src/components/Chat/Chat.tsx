import React, {useState} from 'react';

const Chat: React.FC = () => {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState(new Array<string>());

  const ws = new WebSocket("ws://localhost:5000/ws");
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

  var index = 1;
  return (
    <div>
      <form action="" onSubmit={sendMessage}>
          <input type="text" autoComplete="off" value={message} onChange={handleChange} />
          <button>Send</button>
      </form>
      <ul>
        { messages.map(x => (<li key={index++}>{x}</li>)) }
      </ul>
    </div>
  );
}

export default Chat;
