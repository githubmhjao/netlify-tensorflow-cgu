import ramanData from './data.js';
import ramanUnit from './unit.js'

function getRandomInt(max) {
  return Math.floor(Math.random() * Math.floor(max));
}

async function showExamples(examples, container) {

  const series = examples.map((x, i) => `${i}: A/R = ${ramanData[x].label}`);
  const data = { 
    values: examples.map(x => ramanData[x].profile.map((y, x) => ({x: ramanUnit[x], y: y}))), 
    series 
  }
    
  tfvis.render.linechart(container, data);
}

function getModel() {
  const model = tf.sequential();
  
  const PROFILE_WIDTH = 256;
  
  const encoder = tf.sequential();
  
  // In the first layer of our convolutional neural network we have 
  // to specify the input shape. Then we specify some parameters for 
  // the convolution operation that takes place in this layer.
  encoder.add(tf.layers.dense({inputShape: [PROFILE_WIDTH,], units: 50, useBias: true}));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.  
  encoder.add(tf.layers.dense({units: 10, useBias: true}));
  
  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_LATENT = 2;
  encoder.add(tf.layers.dense({
    units: NUM_LATENT,
    kernelInitializer: 'varianceScaling',
    activation: 'sigmoid'
  }));
  
  const decoder = tf.sequential();
  
  decoder.add(tf.layers.dense({inputShape: [NUM_LATENT,], units: 10, useBias: true }))
  
  decoder.add(tf.layers.dense({units: 50, useBias: true }))
  
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

  return model;
}

function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any 
  // intermediate tensors.
  
  return tf.tidy(() => {
    // Step 1. Shuffle the data    
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.profile)
    const labels = data.map(d => d.label);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 256]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    return [inputTensor, labelTensor]
  });  
}

async function trainModel(model, inputs, labels) {
  const metrics = ['loss', 'val_loss'];
  const fitCallbacks = tfvis.show.fitCallbacks(document.getElementById("container-train"), metrics);
  
  const BATCH_SIZE = 50;
  
  const trainXs = inputs['train']
  const testXs = inputs['test']  
  
  return await model.fit(trainXs, trainXs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testXs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks
  });
}

async function run() {
  const numExamples = 2
  const examples = Array(numExamples).fill(0).map(x => getRandomInt(ramanData.length));
  await showExamples(examples, document.getElementById("container-origin"));
  
  const model = getModel();
  tfvis.show.modelSummary(document.getElementById('container-model'), model);
  
  tf.util.shuffle(ramanData);
  
  const TRAIN_DATA_SIZE = 1000
  const [trainData, trainLabel] = convertToTensor(ramanData.slice(0, TRAIN_DATA_SIZE));
  const [testData, testLabel] = convertToTensor(ramanData.slice(TRAIN_DATA_SIZE))
  
  const inputs = {train: trainData, test: testData}
  const labels = {train: trainLabel, test: testLabel}
  // Train the model  
  await trainModel(model, inputs, labels);
  await showExamples(examples, document.getElementById("container-reconstruct"));
}

document.addEventListener('DOMContentLoaded', run);
