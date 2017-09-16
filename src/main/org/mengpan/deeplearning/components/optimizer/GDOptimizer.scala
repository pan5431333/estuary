package org.mengpan.deeplearning.components.optimizer
import breeze.linalg.DenseMatrix

/**
  * Created by mengpan on 2017/9/9.
  */
object GDOptimizer extends Optimizer with NonHeuristic {

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

