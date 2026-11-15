// MDN loss — port of keras-mdn-layer's `get_mixture_loss_func`, the training
// objective of ../impsy/impsy/mdrnn.py, in TensorFlow.js. The sampling half of
// the MDN is mdnSampler.ts.

import * as tf from "@tensorflow/tfjs-core";

/**
 * Mean negative log-likelihood of `yTrue` under the mixture of
 * diagonal-covariance Gaussians that `yPred` parameterises.
 *
 * @param yTrue targets, shape `(N, D)`
 * @param yPred raw MDN outputs, shape `(N, M·(2D+1))`, laid out
 *   `[mus M×D | sigmas M×D | piLogits M]`; sigmas already positive
 * @returns a scalar; differentiable in `yPred`
 */
export function mdnLoss(
  yTrue: tf.Tensor,
  yPred: tf.Tensor,
  dimension: number,
  numMixtures: number,
): tf.Scalar {
  return tf.tidy(() => {
    const D = dimension;
    const M = numMixtures;
    const total = M * (2 * D + 1);
    const pred = tf.reshape(yPred, [-1, total]);
    const mu = tf.reshape(tf.slice(pred, [0, 0], [-1, M * D]), [-1, M, D]);
    const sigma = tf.reshape(tf.slice(pred, [0, M * D], [-1, M * D]), [-1, M, D]);
    const piLogits = tf.slice(pred, [0, 2 * M * D], [-1, M]);
    const y = tf.reshape(yTrue, [-1, 1, D]);

    // log N(y | mu_k, diag(sigma_k²)) for each mixture k → (N, M)
    const logComponent = tf.sub(
      tf.sub(-0.5 * D * Math.log(2 * Math.PI), tf.sum(tf.log(sigma), -1)),
      tf.mul(0.5, tf.sum(tf.square(tf.div(tf.sub(y, mu), sigma)), -1)),
    );
    const logPi = tf.logSoftmax(piLogits, -1);
    const logProb = tf.logSumExp(tf.add(logPi, logComponent), -1);
    return tf.neg(tf.mean(logProb)) as tf.Scalar;
  });
}
