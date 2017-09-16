package org.mengpan.deeplearning.components.optimizer
import breeze.linalg.DenseMatrix

/**
  * Created by mengpan on 2017/9/9.
  */
class SGDOptimizer extends Optimizer with MiniBatchable with NonHeuristic {
  override def optimize[T <: Seq[DenseMatrix[Double]]](feature: DenseMatrix[Double], label: DenseMatrix[Double])
                                                      (initParams: T)
                                                      (forwardFunc: (DenseMatrix[Double], DenseMatrix[Double], T) => Double)
                                                      (backwardFunc: (DenseMatrix[Double], T) => T): T = {
    (0 until this.iteration).toIterator.foldLeft[T](initParams){
      case (preParams, iterTime) =>
        val minibatches = getMiniBatches(feature, label)
        minibatches.zipWithIndex.foldLeft[T](preParams){
          case (preBatchParams, ((batchFeature, batchLabel), miniBatchTime)) =>
            val cost = forwardFunc(batchFeature, batchLabel, preBatchParams)
            val grads = backwardFunc(batchLabel, preBatchParams)
            updateFunc(preBatchParams, grads)
        }
    }
  }

  private def updateFunc[T <: Seq[DenseMatrix[Double]]](t: T, t1: T): T = {
    val res = for {(param, grad) <- t.zip(t1)} yield (param - learningRate * grad)
    res.asInstanceOf[T]
  }
}

object SGDOptimizer {
  def apply(miniBatchSize: Int): SGDOptimizer = {
    new SGDOptimizer()
      .setMiniBatchSize(miniBatchSize)
  }
}
