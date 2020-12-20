import React from 'react';
import { Graph } from '@visx/network';


const App = (props) => {
  const nodes = [
    {x: 50, y: 20},
    {x: 200, y: 300},
    {x: 300, y: 40}
  ];

  const links = [
    {source: nodes[0], target: nodes[1]},
    {source: nodes[1], target: nodes[2]},
    {source: nodes[2], target: nodes[0]}
  ];

  const graph = {
    nodes, links
  };
  const width = 400;
  const height = 400;
  const background = '#272b4d';

  return (
    <svg width={width} height={height}>
      <rect width={width} height={height} rx={14} fill={background} />
      <Graph graph={graph} />
    </svg>
  )
};

export default App;