import React from "react"
import {useState, useEffect} from "react"

import "./styles.css"
import loadRamanData from "./data.js"
import ramanUnit from "./unit.js"

function getRandomInt(max) {
  return Math.floor(Math.random() * Math.floor(max));
}

async function showExamples(examples, container) {

  const series = examples.map((x, i) => `${i}: A/R = ${x.label}`);
  const data = { 
    values: examples.map(x => x.profile.map((y, i) => ({x: ramanUnit[i], y: y}))), 
    series 
  }
  
  const width = Math.floor(document.documentElement.clientWidth * 0.25)
  const height = Math.floor(document.documentElement.clientHeight * 0.25)
    
  tfvis.render.linechart(container, data, {width: width, height: height});
}

function getModel(PROFILE_WIDTH, numHiddenOne, numHiddenTwo) {
  const model = tf.sequential();
  
  const encoder = tf.sequential({name: "Encoder"});
  
  // In the first layer of our convolutional neural network we have 
  // to specify the input shape. Then we specify some parameters for 
  // the convolution operation that takes place in this layer.
  encoder.add(tf.layers.dense({inputShape: [PROFILE_WIDTH,], units: numHiddenOne, useBias: true}));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.  
  encoder.add(tf.layers.dense({units: numHiddenTwo, useBias: true}));
  
  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_LATENT = 2;
  encoder.add(tf.layers.dense({
    units: NUM_LATENT,
    kernelInitializer: 'varianceScaling',
    activation: 'sigmoid'
  }));
  
  const decoder = tf.sequential({name: "Decoder"});
  
  decoder.add(tf.layers.dense({inputShape: [NUM_LATENT,], units: numHiddenTwo, useBias: true }))
  
  decoder.add(tf.layers.dense({units: numHiddenOne, useBias: true }))
  
  decoder.add(tf.layers.dense({units: PROFILE_WIDTH, useBias: true }))
  
  model.add(encoder);
  model.add(decoder);
  
  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError',
    metrics: ['mse'],
  });

  return {model, encoder, decoder};
}

function convertToTensor(data) {
  
  return tf.tidy(() => {
    // Step 1. Shuffle the data    
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.profile)
    const labels = data.map(d => d.label);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 256]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    return [inputs, inputTensor, labels, labelTensor]
  });  
}

async function trainModel(model, inputs, labels, epochs) {
  const metrics = ['loss', 'val_loss'];
  const fitCallbacks = tfvis.show.fitCallbacks(document.getElementById("container-train"), metrics, {callbacks: ["onEpochEnd"]});
  
  const BATCH_SIZE = 50;
  
  const trainXs = inputs['train']
  const testXs = inputs['test']  
  
  return await model.fit(trainXs, trainXs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testXs],
    epochs: epochs,
    shuffle: true,
    callbacks: fitCallbacks
  });
}

function convertToArray(tensor, label, num, dimension) {
  const data = Array.from(tensor.dataSync())
  // console.log(data)
  return Array(num).fill(0).map((x, i) => ({'profile': data.slice(i * dimension, (i + 1) * dimension), 'label': label[i]}))
}


async function showCodes(codes, container) {
  const seriesLabel = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

  const seriesCode = seriesLabel.map(x => codes.filter(c => c.label == x).map(p => ({x: p.profile[0], y: p.profile[1]})))
  console.log(seriesCode)
  const data = { values: seriesCode, series: seriesLabel.map(x => String(x)) }

  const width = Math.floor(document.documentElement.clientWidth * 0.25)
  const height = Math.floor(document.documentElement.clientHeight * 0.25)
  
  tfvis.render.scatterplot(container, data, {width, height, zoomToFit: true});
}


async function run(numExamples, numHiddenOne, numHiddenTwo, epochs) {
  const NUM_DATASET_ELEMENTS = 1100
  const IMAGE_SIZE = 257
  
  const datasetImages = loadRamanData()
  const ramanData = Array(NUM_DATASET_ELEMENTS)
    .fill(0)
    .map((x, i) => ({'profile': datasetImages.slice(i * IMAGE_SIZE + 1, (i + 1) * IMAGE_SIZE), 'label': Math.round(datasetImages[i * IMAGE_SIZE] * 256)/10 }))
  
  const examples = Array(numExamples).fill(0).map(x => getRandomInt(ramanData.length)).map(x=>ramanData[x]);
  await showExamples(examples, document.getElementById("container-origin"));
  
  const PROFILE_WIDTH = 256
  const {model, encoder, decoder} = getModel(PROFILE_WIDTH, numHiddenOne, numHiddenTwo);
  document.getElementById("container-model").innerHTML = '<div id="container-encoder"></div><div id="container-decoder"></div>'
  tfvis.show.modelSummary(document.getElementById('container-encoder'), encoder);
  tfvis.show.modelSummary(document.getElementById('container-decoder'), decoder);
  
  tf.util.shuffle(ramanData);
  
  const TRAIN_DATA_SIZE = 1000
  const [trainData, trainDataTensor, trainLabel, trainLabelTensor] = convertToTensor(ramanData.slice(0, TRAIN_DATA_SIZE));
  const [testData, testDataTensor, testLabel, testLabelTensor] = convertToTensor(ramanData.slice(TRAIN_DATA_SIZE))
  
  const inputs = {train: trainDataTensor, test: testDataTensor}
  const labels = {train: trainLabelTensor, test: testLabelTensor}
  // Train the model
  document.getElementById("container-train").innerHTML = ""
  await trainModel(model, inputs, labels, epochs);
  
  const [exampleData, exampleDataTensor, exampleLabel, exampleLabelTensor] = convertToTensor(examples)
  const examplePredTensor = model.predict(exampleDataTensor)
  
  const examplePredArray = convertToArray(examplePredTensor, exampleLabel, numExamples, 256)
  await showExamples(examplePredArray, document.getElementById("container-reconstruct"));
  
  const trainCode = convertToArray(encoder.predict(trainDataTensor), trainLabel, TRAIN_DATA_SIZE, 2)
  
  await showCodes(trainCode, document.getElementById("container-latent"))
  
}

function ParameterReadOnly({id, value}) {
  return (
    <div className="unit-control">
        <label for={id} className="unit">{id[0] + id.slice(1)}</label>     
        <input type="text" id={id} className="input-number" value={value} readonly />
    </div>
  )
}

function Parameter({id, value, setValue}) {
  return (
    <div className="unit-control">
        <label for={id} className="unit">{id[0] + id.slice(1)}</label>         
        <input type="number" id={id} className="input-number" value={value} onChange={(e) => {setValue(e.target.value)}} />
    </div>
  )
}

function Parameters({values, setValues}) {
  
  return (
    <div id="container-parameter" className="container">
      <div className="card-header">Parameter</div>
      <div className="card-body">
        <Parameter id="samples" value={values.valueSA} setValue={setValues.setValueSA} />
        <ParameterReadOnly id="dimension" value="256" />
        <Parameter id="firstLayer" value={values.valueFL} setValue={setValues.setValueFL} />
        <Parameter id="secondLayer" value={values.valueSL} setValue={setValues.setValueSL} />
        <ParameterReadOnly id="latent" value="2" />
        <Parameter id="epochs" value={values.valueEP} setValue={setValues.setValueEP} />
      </div>
      <div className="card-footer" id="startTrain" onClick={() => run(values.valueSA, values.valueFL, values.valueSL, values.valueEP)} >Start Train</div>
    </div>
  )
}

function Card({header, id}) {
  return (
    <div className="container">
      <div className="card-header">{header}</div>
      <div className="card-body" id={id}>
        <div className="converter-title">Loading...</div>
      </div>
    </div>
  )
}

export default function APP() {
  const [valueSA, setValueSA] = useState(2)
  const [valueFL, setValueFL] = useState(50)
  const [valueSL, setValueSL] = useState(10)
  const [valueEP, setValueEP] = useState(10)
  
  const values = {valueSA, valueFL, valueSL, valueEP}
  const setValues = {setValueSA, setValueFL, setValueSL, setValueEP}
  
  return (
    <>
      <div className="header"><h1>Raman Spectrum Analyzed by Neural Network AutoEncoder</h1></div>
      <div className="half">
        <Parameters values={values} setValues={setValues} />
        <Card header="Origin Data" id="container-origin" />
        <Card header="Model" id="container-model" />
      </div>
      <div className="half">
        <Card header="Train Metrics" id="container-train" />
        <Card header="Reconstruct Data" id="container-reconstruct" />
        <Card header="Latent Plot" id="container-latent" />
      </div>
    </>)
}
