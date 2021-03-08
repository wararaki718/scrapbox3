import React, {useState} from 'react';


const Chat: React.FC = () => {
  const [item_id, setItemId] = useState("");
  const [token, setToken] = useState("");
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState(new Array<string>());

  var ws: WebSocket;
  const connect = (event: any): void => {
    ws = new WebSocket(`ws://localhost:5000/items/${item_id}/ws?token=${token}`);
    ws.onmessage = (event: any): void => {
      setMessages([...messages, event.data]);
    };
    event.preventDefault();
  };

  const handleChangeItemId = (event: any): void => {
    setItemId(event.target.value);
  };
  const handleChangeMessage = (event: any): void => {
    setMessage(event.target.value);
  };
  const handleChangeToken = (event: any): void => {
    setToken(event.target.value);
  };

  const sendMessage = (event: any): void => {
      ws.send(message);
      setMessage("");
      event.preventDefault();
  };

  return (
    <div>
      <form action="" onSubmit={sendMessage}>
        <label>Item ID: <input type="text" autoComplete="off" value={item_id} onChange={handleChangeItemId} /></label>
        <label>Token: <input type="text" autoComplete="off" value={token} onChange={handleChangeToken} /></label>
        <button onClick={connect}>Connect</button>
        <hr></hr>
        <input type="text" autoComplete="off" value={message} onChange={handleChangeMessage} />
        <button>Send</button>
      </form>
      <ul>
        { messages.map(x => (<li key={Date.now()}>{x}</li>)) }
      </ul>
    </div>
  );
}

export default Chat;
