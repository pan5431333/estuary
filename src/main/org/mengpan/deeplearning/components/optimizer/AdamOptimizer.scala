package org.mengpan.deeplearning.components.optimizer

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{pow, sqrt}
import org.mengpan.deeplearning.components.caseclasses.AdamOptimizationParams
import org.mengpan.deeplearning.components.layers.{DropoutLayer, Layer}
import org.mengpan.deeplearning.utils.{DebugUtils, ResultUtils}

/**
  * Created by mengpan on 2017/9/10.
  */
class AdamOptimizer extends Optimizer{
  override protected var miniBatchSize: Int = _
  protected var momentum: Double = 0.9
  protected var adamParam: Double = 0.999

  def setMiniBatchSize(miniBatchSize: Int): this.type = {
    this.miniBatchSize = miniBatchSize
    this
  }

  def getMiniBatchSize: Int = this.miniBatchSize

  def setMomentumRate(momentum: Double): this.type = {
    this.momentum = momentum
    this
  }

  def setAdamParam(adamRate: Double): this.type = {
    this.adamParam = adamRate
    this
  }

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

            val vW = (this.momentum * momentum._1 + (1.0 - this.momentum) * dw)
            val vB = (this.momentum * momentum._2 + (1.0 - this.momentum) * db)
            val vWCorrected = vW / (1 - pow(this.momentum, miniBatchTime.toInt + 1))
            val vBCorrected = vB / (1 - pow(this.momentum, miniBatchTime.toInt + 1))

            val aW = (this.adamParam * adam._1 + (1.0 - this.adamParam) * pow(dw, 2))
            val aB = (this.adamParam * adam._2 + (1.0 - this.adamParam) * pow(db, 2))
            val aWCorrected = aW / (1 - pow(this.adamParam, miniBatchTime.toInt + 1))
            val aBCorrected = aB / (1 - pow(this.adamParam, miniBatchTime.toInt + 1))

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
