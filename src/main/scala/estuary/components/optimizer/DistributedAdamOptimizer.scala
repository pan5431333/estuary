package estuary.components.optimizer

import breeze.linalg.DenseMatrix
import estuary.components.Exception.GradientExplosionException
import estuary.components.optimizer.AdamOptimizer.AdamParam
import estuary.model.Model
import org.apache.log4j.Logger

class DistributedAdamOptimizer extends AdamOptimizer with AbstractDistributed[AdamParam] {
  override val logger: Logger = Logger.getLogger(this.getClass)

  protected var parameterServer: AdamParam = _

  protected def updateParameterServer(grads: AdamParam, miniBatchTime: Int): Unit = {
    val newParams = updateFunc(parameterServer, grads.modelParam, miniBatchTime)
    this.synchronized(parameterServer = newParams)
  }

  protected def fetchParameterServer(): AdamParam = parameterServer

  def parOptimize(feature: DenseMatrix[Double], label: DenseMatrix[Double], model: Model, initParams: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = {
    parameterServer = AdamParam(initParams, getInitAdam(initParams), getInitAdam(initParams))

    for (i <- (0 until iteration).toIterator) {
      val parBatches = genParBatches(feature, label)
      val modelInstances = parBatches.map(_ => model.copyStructure)

      for {(batch, model) <- parBatches.zip(modelInstances)
           ((feature, label), miniBatchTime) <- batch.zipWithIndex
      } {
        val printMiniBatchUnit = math.max(feature.rows / this.miniBatchSize / 5, 10)
        val params = fetchParameterServer()
        val cost = model.forward(feature, label, params.modelParam)

        if (miniBatchTime % printMiniBatchUnit == 0) {
          logger.info("Iteration: " + i + "|Thread: " + Thread.currentThread().getName + "|=" + "=" * (miniBatchTime / printMiniBatchUnit) + ">> Cost: " + cost)
        }

        //save cost and check for gradient explosion every 10 iterations
        if (miniBatchTime % 10 == 0) {
          addCostHistory(cost)
          try {
            checkGradientsExplosion(cost, minCost)
          } catch {
            case e: GradientExplosionException =>
              handleGradientExplosionException(params, paramSavePath)
              if (exceptionCount > 10) throw e
          }
        }

        val grads = model.backward(label, params.modelParam)
        updateParameterServer(AdamParam(grads, null, null), miniBatchTime)
      }
    }

    parameterServer.modelParam
  }
}

object DistributedAdamOptimizer {
  def apply(miniBatchSize: Int = 64, nTasks: Int = 4): DistributedAdamOptimizer = {
    new DistributedAdamOptimizer()
      .setMiniBatchSize(miniBatchSize)
      .setNTasks(nTasks)
  }
}


