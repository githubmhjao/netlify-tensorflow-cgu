import ramanData from './data.js';
import ramanUnit from './unit.js'

function getRandomInt(max) {
  return Math.floor(Math.random() * Math.floor(max));
}

async function showExamples() {
  // Create a container in the visor
  const surface = document.getElementById("container-surface");  

  // Get the examples
  const numExamples = 2
  const examples = Array(numExamples).fill(0).map(x => getRandomInt(ramanData.length));
  
  const series = examples.map((x, i) => `${i}: A/R = {ramanData[x].label}`);
  const data = { 
    values: examples.map(x => ramanData[x].profile.map((y, x) => {x: ramanUnit[x], y: y})), 
    series 
  }
    
  tfvis.render.linechart(surface, data);
    
}

async function run() {  
  await showExamples();
}

document.addEventListener('DOMContentLoaded', run);
