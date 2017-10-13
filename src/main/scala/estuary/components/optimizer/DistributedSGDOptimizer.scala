package estuary.components.optimizer

import breeze.linalg.DenseMatrix
import estuary.model.Model
import org.apache.log4j.Logger

class DistributedSGDOptimizer extends SGDOptimizer with AbstractDistributed[Seq[DenseMatrix[Double]]] {
  override val logger: Logger = Logger.getLogger(this.getClass)

  protected var parameterServer: Seq[DenseMatrix[Double]] = _

  override def parOptimize(feature: DenseMatrix[Double], label: DenseMatrix[Double], model: Model, initParams: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = {
    parameterServer = initParams

    for (i <- (0 until iteration).toIterator) {
      val parBatches = genParBatches(feature, label)
      val modelInstances = parBatches.map(_ => model.copyStructure)
      for ((batch, model) <- parBatches.zip(modelInstances)) {
        for (((feature, label), miniBatchTime) <- batch.zipWithIndex) {
          val printMiniBatchUnit = math.max(feature.rows / this.miniBatchSize / 5, 10)
          val params = fetchParameterServer()
          val cost = model.forward(feature, label, params)

          if (miniBatchTime % printMiniBatchUnit == 0) {
            logger.info("Iteration: " + i + "|Thread: " + Thread.currentThread().getName + "|=" + "=" * (miniBatchTime / printMiniBatchUnit) + ">> Cost: " + cost)
            addCostHistory(cost)
          }

          val grads = model.backward(label, params)
          updateParameterServer(grads, miniBatchTime)
        }
      }
    }

    parameterServer
  }

  protected def updateParameterServer(grads: Seq[DenseMatrix[Double]], miniBatchTime: Int): Unit = this.synchronized {
    parameterServer = parameterServer.zip(grads).par.map { case (param, grad) =>
      updateFunc(param, grad, miniBatchTime)
    }.seq
  }

  protected def fetchParameterServer(): Seq[DenseMatrix[Double]] = {
    parameterServer
  }

  protected def updateFunc(params: DenseMatrix[Double], grads: DenseMatrix[Double], miniBatchTime: Int): DenseMatrix[Double] =
    updateFunc(List(params), List(grads)).head
}

object DistributedSGDOptimizer {
  def apply(miniBatchSize: Int = 64, nTasks: Int = 4): DistributedSGDOptimizer = {
    new DistributedSGDOptimizer()
      .setMiniBatchSize(miniBatchSize)
      .setNTasks(nTasks)
  }
}


