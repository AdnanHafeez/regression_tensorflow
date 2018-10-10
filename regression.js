//Tensorflow model for regression fitting
const tf = require('@tensorflow/tfjs');

const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));
const learningRate = 0.5;
const optimiser = tf.train.sgd(learningRate);
const numIterations = 75;


function predict(x) {
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(3)))
      .add(b.mul(x.square()))
      .add(c.mul(x))
      .add(d);
  });
}

//console.log(predict(tf.scalar(10)));

function loss(predictions,labels) {
  const meanSquaredError = predictions.sub(labels).square().mean();
  return meanSquaredError;
}


 function train(xs, ys, numIterations = 75) {
  for(let iter = 0; iter < numIterations; iter++) {
    optimiser.minimize( () => {
      const predsYs = predict(xs);
      return loss(predsYs, ys);
    });
  }
}

function learnCoefficients() {
  const trueCoefficients = {a:-0.8, b: -0.2, c:0.9, d:0.5};
  const trainingData = generateData(100, trueCoefficients);

  const predicitionsBefore = predict(trainingData.xs);
  console.log("Before Training...");
  console.log(a.dataSync()[0]);
  console.log(b.dataSync()[0]);
  console.log(c.dataSync()[0]);
  console.log(d.dataSync()[0]);

  train(trainingData.xs, trainingData.ys, numIterations);
  const predicitonsAfter = predict(trainingData.xs);

  console.log("After Training...");
  console.log(a.dataSync()[0]);
  console.log(b.dataSync()[0]);
  console.log(c.dataSync()[0]);
  console.log(d.dataSync()[0]);

}

learnCoefficients();

function generateData(numPoints, coeff, sigma = 0.04) {
  return tf.tidy(() => {
    const [a, b, c, d] = [
      tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
      tf.scalar(coeff.d)
    ];

    const xs = tf.randomUniform([numPoints], -1, 1);

    // Generate polynomial data
    const three = tf.scalar(3, 'int32');
    const ys = a.mul(xs.pow(three))
      .add(b.mul(xs.square()))
      .add(c.mul(xs))
      .add(d)
      // Add random noise to the generated data
      // to make the problem a bit more interesting
      .add(tf.randomNormal([numPoints], 0, sigma));

    // Normalize the y values to the range 0 to 1.
    const ymin = ys.min();
    const ymax = ys.max();
    const yrange = ymax.sub(ymin);
    const ysNormalized = ys.sub(ymin).div(yrange);

    return {
      xs,
      ys: ysNormalized
    };
  })
}
