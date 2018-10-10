//Tensorflow model for regression fitting
const tf = require('@tensorflow/tfjs');

const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));


function predict(x) {
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(3)))
      .add(b.mul(x.square()))
      .add(c.mul(x))
      .add(d);
  });
}

console.log(predict(tf.scalar(10)));

function loss(predicitons,labels) {
  const meanSquaredError = predictions.sub(label).square().mean();
  return meanSquaredError;
}
