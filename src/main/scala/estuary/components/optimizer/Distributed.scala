package estuary.components.optimizer

import breeze.linalg.DenseMatrix
import estuary.model.Model

import scala.collection.parallel.immutable.ParSeq

/**
  * 1. All unimplemented methods are ones that should be implemented in its implementations. These unimplemented methods
  * are also interface methods exposed externally so that external users can invoke them, hence these methods should
  * all be public.
  * 2. All concrete methods are functionality provided to its implementations, which means only implementations are able
  * to use these concrete methods, hence concrete methods should all be protected.
  */
trait Distributed[T] extends Optimizer with MiniBatchable {

  /** Number of models trained in parallel, with sharing the same parameter server. */
  protected val nTasks: Int

  /**
    * Optimize the model in parallel, and returning the trained parameters with the same dimensions of initParams.
    * The method parameter 'model' is used here to create several model instances (with copyStructure() method), and
    * then they are distributed to different threads or machines.
    *
    * @param feature    feature matrix
    * @param label      label matrix with one-hot representation.
    * @param model      an instance of trait Model, used to create many copies and then distribute them to different threads
    *                   or machines.
    * @param initParams initial parameters.
    * @return trained parameters, with same dimension with the given initial parameters.
    */
  def parOptimize(feature: DenseMatrix[Double], label: DenseMatrix[Double], model: Model[T], initParams: T): T

  /**
    * Functionality 1: add cost to MutableList: costHistory with synchronization.
    *
    * @param cost cost value.
    */
  protected def addCostHistory(cost: Double): Unit = this.synchronized {
    costHistory.+=(cost)
    minCost = if (cost < minCost) cost else minCost
  }

  /**
    * Functionality 2: generate parallel minibatches, given a training set (feature, label).
    *
    * @param feature feature matrix
    * @param label   label matrix in one-hot representation
    * @return parallel iterators, each containing several minibatches data set.
    */
  protected def genParBatches(feature: DenseMatrix[Double], label: DenseMatrix[Double]): ParSeq[Iterator[(DenseMatrix[Double], DenseMatrix[Double])]] = {
    val n = feature.rows
    val eachSize = n / nTasks

    for (i <- (0 until nTasks).par) yield {
      val startIndex = i * eachSize
      val endIndex = math.min((i + 1) * eachSize, n)
      getMiniBatches(feature(startIndex until endIndex, ::), label(startIndex until endIndex, ::))
    }
  }


}
