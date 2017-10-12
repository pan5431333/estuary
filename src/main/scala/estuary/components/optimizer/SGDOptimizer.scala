package estuary.components.optimizer

import breeze.linalg.DenseMatrix

/**
  * Stochastic Gradient Descent, i.e. Mini-batch Gradient Descent.
  */
class SGDOptimizer extends Optimizer with MiniBatchable with NonHeuristic {

  /**
    * Implementation of Optimizer.optimize(). Optimizing Machine Learning-like models'
    * parameters on a training dataset (feature, label).
    *
    * @param feature      DenseMatrix of shape (n, p) where n: the number of
    *                     training examples, p: the dimension of input feature.
    * @param label        DenseMatrix of shape (n, q) where n: the number of
    *                     training examples, q: number of distinct labels.
    * @param initParams   Initialized parameters.
    * @param forwardFunc  The cost function.
    *                     inputs: (feature, label, params) of type
    *                     (DenseMatrix[Double], DenseMatrix[Double], T)
    *                     output: cost of type Double.
    * @param backwardFunc A function calculating gradients of all parameters.
    *                     input: (label, params) of type (DenseMatrix[Double], T)
    *                     output: gradients of params of type T.
    * @return Trained parameters.
    */
  override def optimize(feature: DenseMatrix[Double], label: DenseMatrix[Double])
                       (initParams: Seq[DenseMatrix[Double]])
                       (forwardFunc: (DenseMatrix[Double], DenseMatrix[Double], Seq[DenseMatrix[Double]]) => Double)
                       (backwardFunc: (DenseMatrix[Double], Seq[DenseMatrix[Double]]) => Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = {
    val printMiniBatchUnit = feature.rows / this.miniBatchSize / 5 //for each iteration, only print minibatch cost FIVE times.

    (0 until this.iteration).toIterator.foldLeft[Seq[DenseMatrix[Double]]](initParams) { case (preParams, iterTime) =>
      val minibatches = getMiniBatches(feature, label)
      minibatches.zipWithIndex.foldLeft[Seq[DenseMatrix[Double]]](preParams) { case (preBatchParams, ((batchFeature, batchLabel), miniBatchTime)) =>
        val cost = forwardFunc(batchFeature, batchLabel, preBatchParams)
        val grads = backwardFunc(batchLabel, preBatchParams)

        if (miniBatchTime % printMiniBatchUnit == 0)
          logger.info("Iteration: " + iterTime + "|=" + "=" * (miniBatchTime / 10) + ">> Cost: " + cost)
        costHistory.+=(cost)

        updateFunc(preBatchParams, grads)
      }
    }
  }

  /**
    * Update model parameters using Gradient Descent method.
    *
    * @param params Model parameters' values on current iteration.
    * @param grads  Gradients of model parameters on current iteration.
    * @return Updated model parameters.
    */
  protected def updateFunc(params: Seq[DenseMatrix[Double]], grads: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = {
    val res = for {(param, grad) <- params.zip(grads)} yield param - grad * learningRate
    res.asInstanceOf[Seq[DenseMatrix[Double]]]
  }
}

object SGDOptimizer {
  def apply(miniBatchSize: Int): SGDOptimizer = {
    new SGDOptimizer()
      .setMiniBatchSize(miniBatchSize)
  }
}
