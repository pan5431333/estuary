package org.mengpan.deeplearning.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.log4j.Logger
import org.mengpan.deeplearning.utils.ResultUtils.{BackwardRes, ForwardRes}
import org.mengpan.deeplearning.utils.{ActivationUtils, DebugUtils, GradientUtils}

/**
  * Created by mengpan on 2017/8/26.
  */
trait Layer{
  private val logger = Logger.getLogger("Layer")

  var numHiddenUnits: Int
  var activationFunc: Byte


  def setNumHiddenUnits(numHiddenUnits: Int): this.type = {
    this.numHiddenUnits = numHiddenUnits
    this
  }

  def forward(yPrevious: DenseMatrix[Double], w: DenseMatrix[Double],
              b: DenseVector[Double]): ForwardRes = {
    val numExamples = yPrevious.rows
    logger.debug(DebugUtils.matrixShape(yPrevious, "yPrevious"))
    logger.debug(DebugUtils.matrixShape(w, "w"))
    logger.debug(DebugUtils.vectorShape(b, "b"))
    val zCurrent = yPrevious * w + DenseVector.ones[Double](numExamples) * b.t
    val yCurrent = ActivationUtils.getActivationFunc(this.activationFunc)(zCurrent)
    logger.debug("yCurrent: " + yCurrent)
    ForwardRes(yPrevious, zCurrent, yCurrent)
  }

  def backward(dYCurrent: DenseMatrix[Double], forwardRes: ForwardRes,
               w: DenseMatrix[Double], b: DenseVector[Double]): BackwardRes = {
    val numExamples = dYCurrent.rows

    val yPrevious = forwardRes.yPrevious
    val zCurrent = forwardRes.zCurrent
    val yCurrent = forwardRes.yCurrent

    val dZCurrent = dYCurrent *:*
      GradientUtils.getGradByFuncType(this.activationFunc)(zCurrent)

    val dWCurrent = yPrevious.t * dZCurrent / numExamples.toDouble
    val dBCurrent = (DenseVector.ones[Double](numExamples).t * dZCurrent).t /
      numExamples.toDouble
    val dYPrevious = dZCurrent * w.t

    BackwardRes(dYPrevious, dWCurrent, dBCurrent)
  }

  override def toString: String = super.toString
}
