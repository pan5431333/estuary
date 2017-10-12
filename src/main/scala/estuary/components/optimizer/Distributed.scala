package estuary.components.optimizer

import breeze.linalg.DenseMatrix
import estuary.model.Model

import scala.collection.parallel.immutable.ParSeq

trait Distributed extends Optimizer with MiniBatchable {

  protected var nTasks: Int = 4

  def parOptimize(feature: DenseMatrix[Double], label: DenseMatrix[Double], model: Model, initParams: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]]

  def setNTasks(nTasks: Int): this.type = {
    this.nTasks = nTasks
    this
  }

  protected def genParBatches(feature: DenseMatrix[Double], label: DenseMatrix[Double], nTasks: Int): ParSeq[Iterator[(DenseMatrix[Double], DenseMatrix[Double])]] = {
    val n = feature.rows
    val eachSize = n / nTasks

    for (i <- (0 until nTasks).par) yield {
      val startIndex = i * eachSize
      val endIndex = math.min((i + 1) * eachSize, n)
      getMiniBatches(feature(startIndex until endIndex, ::), label(startIndex until endIndex, ::))
    }
  }


}
