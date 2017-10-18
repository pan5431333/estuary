package estuary.components.optimizer

import breeze.linalg.DenseMatrix
import breeze.numerics.{pow, sqrt}
import estuary.components.optimizer.AdamOptimizer.AdamParam
import org.apache.log4j.Logger

/**
  * Adam Optimizer, a very efficient and recommended optimizer for Deep Neural Network.
  */
class AdamOptimizer(val iteration: Int,
                    val learningRate: Double,
                    val paramSavePath: String,
                    val miniBatchSize: Int,
                    val momentumRate: Double,
                    val adamRate: Double) extends Optimizer with MiniBatchable with Heuristic {
  protected val logger: Logger = Logger.getLogger(this.getClass)

  override def optimize(feature: DenseMatrix[Double], label: DenseMatrix[Double])
                       (initParams: Seq[DenseMatrix[Double]])
                       (forwardFunc: (DenseMatrix[Double], DenseMatrix[Double], Seq[DenseMatrix[Double]]) => Double)
                       (backwardFunc: (DenseMatrix[Double], Seq[DenseMatrix[Double]]) => Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = {
    val initMomentum = getInitAdam(initParams)
    val initAdam = getInitAdam(initParams)
    val initAdamParam = AdamParam(initParams, initMomentum, initAdam)
    val numExamples = feature.rows
    val printMiniBatchUnit = numExamples / this.miniBatchSize / 5 //for each iteration, only print minibatch cost FIVE times.

    (0 to this.iteration).foldLeft[AdamParam](initAdamParam) { case (preParam, iterTime) =>
      val minibatches = getMiniBatches(feature, label)
      minibatches.zipWithIndex.foldLeft[AdamParam](preParam) { case (preBatchParams, ((batchFeature, batchLabel), miniBatchTime)) =>
        val cost = forwardFunc(batchFeature, batchLabel, preBatchParams.modelParam)
        val grads = backwardFunc(batchLabel, preBatchParams.modelParam)

        MiniBatchable.printCostInfo(cost, iterTime, miniBatchTime, printMiniBatchUnit, logger)
        addCostHistory(cost)

        updateFunc(preBatchParams, grads, iterTime * miniBatchSize + miniBatchTime)
      }
    }.modelParam
  }

  protected def handleGradientExplosionException(params: Any, paramSavePath: String): Unit = {
    exceptionCount += 1
    saveDenseMatricesToDisk(params.asInstanceOf[AdamParam].modelParam, paramSavePath)
  }

  /**
    * Initialize momentum or RMSProp parameters to all zeros.
    */
  protected def getInitAdam(modelParams: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = {
    modelParams.par.map { m => DenseMatrix.zeros[Double](m.rows, m.cols) }.seq
  }

  /**
    * Update model parameters, momentum parameters and RMSProp parameters by Adam method.
    *
    * @param params        model parameters, momentum parameters and RMSProp parameters, where T is the type of model parameters.
    * @param grads         Gradients of model parameters on current iteration.
    * @param miniBatchTime Current minibatch time.
    * @return Updated model parameters, momentum parameters and RMSProp parameters.
    */
  def updateFunc(params: AdamParam, grads: Seq[DenseMatrix[Double]], miniBatchTime: Int): AdamParam = {
    val (ps, vs, ws) = params match {
      case AdamParam(a, b, c) => (a, b, c)
    }

    val updatedParams = ps.zip(vs).zip(ws).zip(grads).par.map { case (((p, v), a), grad) =>
      val vN = v * this.momentumRate + grad * (1.0 - this.momentumRate)
      val vNCorrected = vN / (1.0 - pow(this.momentumRate, miniBatchTime.toDouble + 1.0))

      val aN = a * this.adamRate + (1.0 - this.adamRate) * pow(grad, 2.0)
      val aNCorrected = aN / (1.0 - pow(this.adamRate, miniBatchTime.toDouble + 1.0))

      val pN = p - learningRate * vNCorrected /:/ (sqrt(aNCorrected) + 1E-8)
      (pN, vN, aN)
    }.seq.toList

    val (modelParams, momentumAndAdam) = updatedParams.unzip(f => (f._1, (f._2, f._3)))

    val (momentum, adam) = momentumAndAdam.unzip(f => (f._1, f._2))

    AdamParam(modelParams.asInstanceOf[Seq[DenseMatrix[Double]]], momentum.asInstanceOf[Seq[DenseMatrix[Double]]], adam.asInstanceOf[Seq[DenseMatrix[Double]]])
  }
}

object AdamOptimizer {

  case class AdamParam(modelParam: Seq[DenseMatrix[Double]], momentumParam: Seq[DenseMatrix[Double]], adamParam: Seq[DenseMatrix[Double]])

  def apply(iteration: Int = 100, learningRate: Double = 0.001, paramSavePath: String = System.getProperty("user.dir"), miniBatchSize: Int = 64, momentumRate: Double = 0.9, adamRate: Double = 0.999): AdamOptimizer = {
    new AdamOptimizer(iteration, learningRate, paramSavePath, miniBatchSize, momentumRate, adamRate)
  }
}
