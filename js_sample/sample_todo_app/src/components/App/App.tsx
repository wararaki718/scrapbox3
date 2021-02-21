import React, { useState } from 'react';
import Todo from '../Todo';

const App: React.FC = () => {
  const [todos, setTodos] = useState(new Array<string>());

  const newTodo = (): void => {
    setTodos([...todos, '']);
  };

  const clearTodo = (): void => {
    setTodos(new Array<string>());
  };

  return (
    <div>
      <div>
        <button onClick={newTodo}>New</button>
        <button onClick={clearTodo}>Clear</button>
      </div>
      <div>
        { todos.map(x => <Todo />) }
      </div>
    </div>
  );
}

export default App;
