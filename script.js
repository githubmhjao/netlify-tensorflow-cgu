import ramanData from "./data.js"
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
    
  tfvis.render.linechart(container, data);
}

function getModel(PROFILE_WIDTH, numHiddenOne, numHiddenTwo) {
  const model = tf.sequential();
  
  const encoder = tf.sequential();
  
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
  
  const decoder = tf.sequential();
  
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
  const fitCallbacks = tfvis.show.fitCallbacks(document.getElementById("container-train"), metrics);
  
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
  console.log(data)
  return Array(num).fill(0).map((x, i) => ({'profile': data.slice(i * dimension, (i + 1) * dimension), 'label': label[i]}))
}


async function showCodes(codes, container) {
  const seriesLabel = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

  const seriesCode = seriesLabel.map(x => codes.filter(c => c.label == x).map(p => ({x: p.profile[0], y: p.profile[1]})))
  console.log(seriesCode)
  const data = { values: seriesCode, series: seriesLabel.map(x => String(x)) }

  tfvis.render.scatterplot(container, data);
}


async function run(numExamples, numHiddenOne, numHiddenTwo, epochs) {
  
  const examples = Array(numExamples).fill(0).map(x => getRandomInt(ramanData.length)).map(x=>ramanData[x]);
  await showExamples(examples, document.getElementById("container-origin"));
  
  const PROFILE_WIDTH = 256
  const {model, encoder} = getModel(PROFILE_WIDTH, numHiddenOne, numHiddenTwo);
  tfvis.show.modelSummary(document.getElementById('container-model'), model);
  
  tf.util.shuffle(ramanData);
  
  const TRAIN_DATA_SIZE = 1000
  const [trainData, trainDataTensor, trainLabel, trainLabelTensor] = convertToTensor(ramanData.slice(0, TRAIN_DATA_SIZE));
  const [testData, testDataTensor, testLabel, testLabelTensor] = convertToTensor(ramanData.slice(TRAIN_DATA_SIZE))
  
  const inputs = {train: trainDataTensor, test: testDataTensor}
  const labels = {train: trainLabelTensor, test: testLabelTensor}
  // Train the model  
  await trainModel(model, inputs, labels, epochs);
  
  const [exampleData, exampleDataTensor, exampleLabel, exampleLabelTensor] = convertToTensor(examples)
  const examplePredTensor = model.predict(exampleDataTensor)
  
  const examplePredArray = convertToArray(examplePredTensor, exampleLabel, numExamples, 256)
  await showExamples(examplePredArray, document.getElementById("container-reconstruct"));
  
  const trainCode = convertToArray(encoder.predict(trainDataTensor), trainLabel, TRAIN_DATA_SIZE, 2)
  
  await showCodes(trainCode, document.getElementById("container-code"))
    
}

document.addEventListener('DOMContentLoaded', run(2, 50, 10, 10));
