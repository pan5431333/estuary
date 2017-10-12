package estuary.components.optimizer

import breeze.linalg.DenseMatrix
import estuary.model.Model

class DistributedSGDOptimizer extends SGDOptimizer with MiniBatchable with Distributed {

  protected var parameterServer: Seq[DenseMatrix[Double]] = _

  override def parOptimize(feature: DenseMatrix[Double], label: DenseMatrix[Double], model: Model, initParams: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]]  = {
    parameterServer = initParams

    for (i <- 0 until iteration) {
      val parBatches = genParBatches(feature, label, nTasks)
      val modelInstances = parBatches.map(_ => model.copyStructure)
      for ((batch, model) <- parBatches.zip(modelInstances)) {
        for (((feature, label), miniBatchTime) <- batch.zipWithIndex) {
          //          val printMiniBatchUnit = feature.rows / this.miniBatchSize / 5
          val printMiniBatchUnit = 10
          val params = fetchParameterServer()
          val cost = model.forward(feature, label, params)

          if (miniBatchTime % printMiniBatchUnit == 0)
            logger.info("Iteration: " + i + "|=" + "=" * (miniBatchTime / printMiniBatchUnit) + ">> Cost: " + cost)
          costHistory.+=(cost)

          val grads = model.backward(label, params)
          updateParameterServer(grads, miniBatchTime)
        }
      }
    }

    parameterServer
  }

  protected def updateParameterServer(grads: Seq[DenseMatrix[Double]], miniBatchTime: Int): Unit = {
    parameterServer = parameterServer.zip(grads).par.map { case (param, grad) =>
      updateFunc(param, grad, miniBatchTime)
    }.seq
  }

  protected def updateFunc(params: DenseMatrix[Double], grads: DenseMatrix[Double], miniBatchTime: Int): DenseMatrix[Double] =
    updateFunc(List(params), List(grads)).head

  protected def fetchParameterServer(): Seq[DenseMatrix[Double]] = {
    parameterServer
  }
}

object DistributedSGDOptimizer {
  def apply(miniBatchSize: Int = 64, nTasks: Int = 4): DistributedSGDOptimizer = {
    new DistributedSGDOptimizer()
      .setMiniBatchSize(miniBatchSize)
      .setNTasks(nTasks)
  }
}


