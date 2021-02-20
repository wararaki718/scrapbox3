import React, {useState} from 'react';


const Todo: React.FC = () => {
    const [task, setTask] = useState('');
    const [display, setDisplay] = useState({display: 'block'});
    const handleChange = (event: any) => {
        setTask(event.target.value);
    };

    const deleteTodo = ():void => {
        setDisplay({display:'none'});
    };
    
    return (
        <div style={display}>
            <input type="checkbox" />
            <input
                type="text"
                placeholder="input your task..."
                value={task}
                onChange={handleChange}
            />
            <button onClick={deleteTodo}>Delete</button>
        </div>
    );
}

export default Todo;
