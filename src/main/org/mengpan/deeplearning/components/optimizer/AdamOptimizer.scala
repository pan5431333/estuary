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

  def updateParams(paramsList: List[(DenseMatrix[Double], DenseVector[Double])],
                   previousMomentum: List[(DenseMatrix[Double], DenseVector[Double])],
                   previousAdam: List[(DenseMatrix[Double], DenseVector[Double])],
                   learningrate: Double,
                   backwardResList: List[ResultUtils.BackwardRes],
                   iterationTime: Int,
                   miniBatchTime: Double,
                   layers: List[Layer]): AdamOptimizationParams = {

    val updatedParams = paramsList
      .zip(backwardResList)
      .zip(layers)
      .zip(previousMomentum)
      .zip(previousAdam)
      .map{f =>
        val layer = f._1._1._2
        val (w, b) = f._1._1._1._1

        val backwardRes = f._1._1._1._2
        val momentum = f._1._2
        val adam = f._2

        layer match {
          case _:DropoutLayer => ((w, b), momentum, adam)
          case _ =>
            val dw = backwardRes.dWCurrent
            val db = backwardRes.dBCurrent

            val vW = (this.momentumRate * momentum._1 + (1.0 - this.momentumRate) * dw)
            val vB = (this.momentumRate * momentum._2 + (1.0 - this.momentumRate) * db)
            val vWCorrected = vW / (1 - pow(this.momentumRate, miniBatchTime.toInt + 1))
            val vBCorrected = vB / (1 - pow(this.momentumRate, miniBatchTime.toInt + 1))

            val aW = (this.adamRate * adam._1 + (1.0 - this.adamRate) * pow(dw, 2))
            val aB = (this.adamRate * adam._2 + (1.0 - this.adamRate) * pow(db, 2))
            val aWCorrected = aW / (1 - pow(this.adamRate, miniBatchTime.toInt + 1))
            val aBCorrected = aB / (1 - pow(this.adamRate, miniBatchTime.toInt + 1))

            logger.debug(DebugUtils.matrixShape(w, "w"))
            logger.debug(DebugUtils.matrixShape(dw, "dw"))

            w :-= learningrate * vWCorrected /:/ (sqrt(aWCorrected) + 1E-8)
            b :-= learningrate * vBCorrected /:/ (sqrt(aBCorrected) + 1E-8)

            ((w, b), (vW, vB), (aW, aB))
        }
      }

    val (modelParams, momentumAndAdam) = updatedParams
      .unzip(f => (f._1, (f._2, f._3)))

    val (momentum, adam) = momentumAndAdam.unzip(f => (f._1, f._2))

    AdamOptimizationParams(modelParams, momentum, adam)
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
