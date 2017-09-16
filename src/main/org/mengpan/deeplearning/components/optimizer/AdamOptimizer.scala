package org.mengpan.deeplearning.components.optimizer

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{pow, sqrt}
import org.mengpan.deeplearning.components.caseclasses.AdamOptimizationParams
import org.mengpan.deeplearning.components.layers.{DropoutLayer, Layer}
import org.mengpan.deeplearning.utils.{DebugUtils, ResultUtils}

/**
  * Created by mengpan on 2017/9/10.
  */
class AdamOptimizer extends Optimizer with MiniBatchable with Heuristic{
  protected var momentumRate: Double = 0.9
  protected var adamRate: Double = 0.999

  case class AdamParam[T <: Seq[DenseMatrix[Double]]](modelParam: T, momentumParam: T, adamParam: T)

  override def optimize[T <: Seq[DenseMatrix[Double]]](feature: DenseMatrix[Double], label: DenseMatrix[Double])
                                                      (initParams: T)
                                                      (forwardFunc: (DenseMatrix[Double], DenseMatrix[Double], T) => Double)
                                                      (backwardFunc: (DenseMatrix[Double], T) => T): T = {
    val initMomentum = initMomentumOrAdam(initParams)
    val initAdam = initMomentumOrAdam(initParams)
    val initAdamParam = AdamParam[T](initParams, initMomentum, initAdam)
    val numExamples = feature.rows
    val printMiniBatchUnit = ((numExamples / this.getMiniBatchSize).toInt / 5).toInt //for each iteration, only print minibatch cost FIVE times.

    (0 until this.iteration).toIterator.foldLeft[AdamParam[T]](initAdamParam){
      case (preParam, iterTime) =>
        val minibatches = getMiniBatches(feature, label)
        minibatches.zipWithIndex.foldLeft[AdamParam[T]](preParam){
          case (preBatchParams, ((batchFeature, batchLabel), miniBatchTime)) =>
            val cost = forwardFunc(batchFeature, batchLabel, preBatchParams.modelParam)
            val grads = backwardFunc(batchLabel, preBatchParams.modelParam)

            if (miniBatchTime % printMiniBatchUnit == 0)
              logger.info("Iteration: " + iterTime + "|=" + "=" * (miniBatchTime / 10) + ">> Cost: " + cost)
            costHistory.+=(cost)

            updateFunc(preBatchParams, grads, miniBatchTime)
        }
    }.modelParam
  }

  private def initMomentumOrAdam[T <: Seq[DenseMatrix[Double]]](t: T): T = {
    t.map{m =>
      DenseMatrix.zeros[Double](m.rows, m.cols)
    }.asInstanceOf[T]
  }

  private def updateFunc[T <: Seq[DenseMatrix[Double]]](value: AdamParam[T], grads: T, miniBatchTime: Int): AdamParam[T] = {
    val (ps, vs, ws) = value match {
      case AdamParam(a, b, c) => (a, b, c)
    }

    val updatedParams = ps.zip(vs).zip(ws).zip(grads).map{
      case (((p, v), a), grad) =>
        val vN = (this.momentumRate * v + (1.0 - this.momentumRate) * grad)
        val vNCorrected = vN / (1 - pow(this.momentumRate, miniBatchTime.toInt + 1))

        val aN = (this.adamRate * a + (1.0 - this.adamRate) * pow(grad, 2))
        val aNCorrected = aN / (1 - pow(this.adamRate, miniBatchTime.toInt + 1))

        val pN = p - learningRate * vNCorrected /:/ (sqrt(aNCorrected) + 1E-8)
        (pN, vN, aN)
    }

    val (modelParams, momentumAndAdam) = updatedParams
      .unzip(f => (f._1, (f._2, f._3)))

    val (momentum, adam) = momentumAndAdam.unzip(f => (f._1, f._2))

    AdamParam[T](modelParams.asInstanceOf[T], momentum.asInstanceOf[T], adam.asInstanceOf[T])
  }
}

object AdamOptimizer{
  def apply(miniBatchSize: Int = 64, momentumRate: Double = 0.9, adamRate: Double = 0.999): AdamOptimizer = {
    new AdamOptimizer()
      .setMiniBatchSize(miniBatchSize)
      .setAdamRate(adamRate)
      .setMomentumRate(momentumRate)
  }
}
