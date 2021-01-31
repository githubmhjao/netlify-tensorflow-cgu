const RAMAN_IMAGE_PATH = "https://i.imgur.com/tPsSRrS.png"
const IMAGE_SIZE = 257
const NUM_DATASET_ELEMENTS = 1100

async function loadRamanData() {

  let datasetImages
  const img = new Image()
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')

  const imgRequest = new Promise((resolve, reject) => {
    img.crossOrigin = '';
    img.onload = () => {
      img.width = img.naturalWidth;
      img.height = img.naturalHeight;

      const datasetBytesBuffer =
          new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

      const chunkSize = 100;
      canvas.width = img.width;
      canvas.height = chunkSize;

      for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
        const datasetBytesView = new Float32Array(
            datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize);
        ctx.drawImage(
            img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
            chunkSize);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        for (let j = 0; j < imageData.data.length / 4; j++) {
          // All channels hold an equal value since the image is grayscale, so
          // just read the red channel.
          datasetBytesView[j] = imageData.data[j * 4] / 255;
        }
      }
      datasetImages = new Float32Array(datasetBytesBuffer);
  
      resolve();
    };
    img.src = RAMAN_IMAGE_PATH;
  });

  await Promise.all([imgRequest])

  return datasetImages
}

export default loadRamanData
