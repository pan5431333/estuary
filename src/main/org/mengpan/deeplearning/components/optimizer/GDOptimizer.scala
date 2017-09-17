package org.mengpan.deeplearning.components.optimizer
import breeze.linalg.DenseMatrix

/**
  * Gradient Descent optimizer.
  */
object GDOptimizer extends Optimizer with NonHeuristic {

  /**
    * Implementation of Optimizer.optimize().
    * Optimizing Machine Learning-like models' parameters on a training
    * dataset (feature, label).
    * @param feature DenseMatrix of shape (n, p) where n: the number of
    *                training examples, p: the dimension of input feature.
    * @param label DenseMatrix of shape (n, q) where n: the number of
    *              training examples, q: number of distinct labels.
    * @param initParams Initialized parameters.
    * @param forwardFunc The cost function.
    *                    inputs: (feature, label, params) of type
    *                           (DenseMatrix[Double], DenseMatrix[Double], T)
    *                    output: cost of type Double.
    * @param backwardFunc A function calculating gradients of all parameters.
    *                     input: (label, params) of type (DenseMatrix[Double], T)
    *                     output: gradients of params of type T.
    * @tparam T The type of model parameters.
    *           For Neural Network, T is List[DenseMatrix[Double]]
    * @return Trained parameters.
    */
  override def optimize[T <: Seq[DenseMatrix[Double]]](feature: DenseMatrix[Double], label: DenseMatrix[Double])
                          (initParams: T)
                          (forwardFunc: (DenseMatrix[Double], DenseMatrix[Double], T) => Double)
                          (backwardFunc: (DenseMatrix[Double], T) => T): T = {
    (0 until this.iteration).foldLeft[T](initParams){
      case (preParams, iterTime) =>
        val cost = forwardFunc(feature, label, preParams)
        val grads = backwardFunc(label, preParams)
        updateFunc(preParams, grads)
    }
  }

  private def updateFunc[T <: Seq[DenseMatrix[Double]]](params: T, grads: T): T = {
    val res = for {(param, grad) <- params.zip(grads)} yield (param - learningRate * grad)
    res.asInstanceOf[T]
  }
}

