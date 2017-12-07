package estuary.components.optimizer

import akka.event.LoggingAdapter
import breeze.linalg.DenseMatrix
import estuary.model.Model
import org.slf4j.Logger

import scala.collection.parallel.immutable.ParSeq

/**
  * 1. All unimplemented methods are ones that should be implemented in its implementations. These unimplemented methods
  * are also interface methods exposed externally so that external users can invoke them, hence these methods should
  * all be public.
  * 2. All concrete methods are functionality provided to its implementations, which means only implementations are able
  * to use these concrete methods, hence concrete methods should all be protected.
  *
  * @note Some lessons learned from building large scala distributed neural network:
  *       1. The right way to slice parallel data sets from training set:
  *       1) Split up training set into a ParSeq of multiple parallel data sets (without shuffle) first
  *       2) For each parallel data set, do sequential optimization (with sharing of parameter server)
  *      2. The right way to update parameter server:
  *       After calculating gradients, before updating parameter server, remember to fetch current parameters from
  *       parameter server again, since parameter server might be updated by other threads during the process of
  *       calculating gradients in current thread.
  */
trait ParallelOptimizer[ModelParam] extends Optimizer with MiniBatchable {

  /** Number of models trained in parallel, with sharing the same parameter server. */
  protected val nTasks: Int

  /**
    * Optimize the model in parallel, and returning the trained parameters with the same dimensions of initParams.
    * The method parameter 'model' is used here to create several model instances (with copyStructure() method), and
    * then they are distributed to different threads or machines.
    *
    * @param feature    feature matrix
    * @param label      label matrix with one-hot representation.
    * @param model      an instance of trait ModelLike, used to create many copies and then distribute them to different threads
    *                   or machines.
    * @param initParams initial parameters.
    * @return trained parameters, with same dimension with the given initial parameters.
    */
  def parOptimize(feature: DenseMatrix[Double], label: DenseMatrix[Double], model: Model[ModelParam], initParams: ModelParam): ModelParam

  /**
    * Functionality 1: add cost to MutableList: costHistory with synchronization.
    *
    * @param cost cost value.
    */
  override protected def addCostHistory(cost: Double): Unit = this.synchronized {
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
  protected def genParBatches(feature: DenseMatrix[Double], label: DenseMatrix[Double]): ParSeq[(DenseMatrix[Double], DenseMatrix[Double])] = {
    val n = feature.rows
    val eachSize = n / nTasks

    for (i <- (0 until nTasks).par) yield {
      val startIndex = i * eachSize
      val endIndex = math.min((i + 1) * eachSize, n)
      (feature(startIndex until endIndex, ::), label(startIndex until endIndex, ::))
    }
  }


}

object ParallelOptimizer {
  def printCostInfo(cost: Double, iterTime: Int, miniBatchTime: Int, printCostUnit: Int, logger: Logger): Unit = {
    if (miniBatchTime % printCostUnit == 0) {
      logger.info("Iteration: " + iterTime + "|Thread: " + Thread.currentThread().getName + "|=" + "=" * (miniBatchTime / printCostUnit) + ">> Cost: " + cost)
    }
  }

  def printCostInfo(cost: Double, iterTime: Int, miniBatchTime: Int, printCostUnit: Int, logger: LoggingAdapter): Unit = {
    if (miniBatchTime % printCostUnit == 0) {
      logger.info("Iteration: " + iterTime + "|Thread: " + Thread.currentThread().getName + "|=" + "=" * (miniBatchTime / printCostUnit) + ">> Cost: " + cost)
    }
  }

  def printAkkaCostInfo(cost: Double, iterTime: Int, miniBatchTime: Int, printCostUnit: Int, logger: LoggingAdapter): Unit = {
    if (miniBatchTime % printCostUnit == 0) {
      logger.info("Iteration: " + iterTime + "|=" + "=" * (miniBatchTime / printCostUnit) + ">> Cost: " + cost)
    }
  }
}
