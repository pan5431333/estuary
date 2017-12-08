package estuary.components.optimizer

import breeze.linalg.DenseMatrix
import estuary.components.Exception.GradientExplosionException
import estuary.components.optimizer.AdamOptimizer.AdamParam
import estuary.model.{Model, ModelLike}
import org.slf4j.{Logger, LoggerFactory}

class AdamParallelOptimizer(override val iteration: Int,
                            override val learningRate: Double,
                            override val paramSavePath: String,
                            override val miniBatchSize: Int,
                            override val momentumRate: Double,
                            override val adamRate: Double,
                            val nTasks: Int)
  extends AdamOptimizer(iteration, learningRate, paramSavePath, miniBatchSize, momentumRate, adamRate)
    with AbstractParallelOptimizer[AdamParam, Seq[DenseMatrix[Double]]] {
  override val logger: Logger = LoggerFactory.getLogger(this.getClass)

  protected def updateParameterServer(grads: AdamParam, miniBatchTime: Int): Unit = {
    val nowParams = fetchParameterServer()
    val newParams = updateFunc(nowParams, grads.modelParam, miniBatchTime)
    updateParameterServer(newParams)
  }

  def parOptimize(feature: DenseMatrix[Double], label: DenseMatrix[Double], model: Model, initParams: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = {
    updateParameterServer(AdamParam(initParams, getInitAdam(initParams), getInitAdam(initParams)))

    val parBatches = genParBatches(feature, label)
    val modelInstances = parBatches.map(_ => model.copyStructure)

    for {(batch, model) <- parBatches.zip(modelInstances)
         i <- 0 until iteration
         ((feature, label), miniBatchTime) <- getMiniBatches(batch._1, batch._2).zipWithIndex
    } {
      val printMiniBatchUnit = math.max(feature.rows / this.miniBatchSize / 5, 10)
      val params = fetchParameterServer()
      val cost = model.forwardAndCalCost(feature, label, params.modelParam)

      ParallelOptimizer.printCostInfo(cost, i, miniBatchTime, printMiniBatchUnit, logger)

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

      val grads = model.backwardWithGivenParams(label, params.modelParam)

      updateParameterServer(AdamParam(grads, null, null), miniBatchTime)
    }
    fetchParameterServer().modelParam
  }
}

object AdamParallelOptimizer {
  def apply(iteration: Int = 100, learningRate: Double = 0.001, paramSavePath: String = System.getProperty("user.dir"), miniBatchSize: Int = 64, momentumRate: Double = 0.9, adamRate: Double = 0.999, nTasks: Int = 4): AdamParallelOptimizer = {
    new AdamParallelOptimizer(iteration, learningRate, paramSavePath, miniBatchSize, momentumRate, adamRate, nTasks)
  }
}


