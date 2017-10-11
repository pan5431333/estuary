package estuary.components.optimizer

import breeze.linalg.DenseMatrix

/**
  * Gradient Descent optimizer.
  */
object GDOptimizer extends Optimizer with NonHeuristic {

  /**
    * Implementation of Optimizer.optimize().
    * Optimizing Machine Learning-like models' parameters on a training
    * dataset (feature, label).
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
    * @tparam T The type of model parameters.
    * @return Trained parameters.
    */
  override def optimize[T <: DenseMatrix[Double]](feature: DenseMatrix[Double], label: DenseMatrix[Double])
                                                      (initParams: Seq[T])
                                                      (forwardFunc: (DenseMatrix[Double], DenseMatrix[Double], Seq[T]) => Double)
                                                      (backwardFunc: (DenseMatrix[Double], Seq[T]) => Seq[T]): Seq[T] = {
    (0 until this.iteration).foldLeft[Seq[T]](initParams) { case (preParams, iterTime) =>
      val cost = forwardFunc(feature, label, preParams)
      val grads = backwardFunc(label, preParams)
      updateFunc(preParams, grads)
    }
  }

  private def updateFunc[T <: DenseMatrix[Double]](params: Seq[T], grads: Seq[T]): Seq[T] = {
    val res = for {(param, grad) <- params.zip(grads)} yield param - grad * learningRate
    res.asInstanceOf[Seq[T]]
  }
}

