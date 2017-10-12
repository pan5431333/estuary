package estuary.components.optimizer
import breeze.linalg.DenseMatrix
import estuary.model.Model

class DistributedAdamOptimizer extends AdamOptimizer with MiniBatchable with Distributed {

  override def parOptimize(feature: DenseMatrix[Double], label: DenseMatrix[Double], model: Model, initParams: Seq[DenseMatrix[Double]]) = ???
}
