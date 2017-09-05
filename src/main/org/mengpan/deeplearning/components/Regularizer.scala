package org.mengpan.deeplearning.components

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{log, pow}
import org.mengpan.deeplearning.layers.Layer
import org.mengpan.deeplearning.utils.ResultUtils.{BackwardRes, ForwardRes}

/**
  * Created by mengpan on 2017/9/5.
  */
trait Regularizer {
  var lambda: Double = _

  def setLambda(lambda: Double): this.type = {
    if (lambda < 0) throw new IllegalArgumentException("Lambda must be nonnegative!")

    this.lambda = lambda
    this
  }

  protected def getReguCost(paramsList: List[(DenseMatrix[Double], DenseVector[Double])]): Double

  def calCost(predicted: DenseVector[Double], label: DenseVector[Double],
              paramsList: List[(DenseMatrix[Double], DenseVector[Double])]):
  Double= {
    val originalCost = -(label.t * log(predicted + pow(10.0, -9)) + (1.0 - label).t * log(1.0 - predicted + pow(10.0, -9))) / label.length.toDouble
    val reguCost = getReguCost(paramsList)

    originalCost + this.lambda * reguCost / label.length.toDouble
  }


  def backward(feature: DenseMatrix[Double], label: DenseVector[Double],
               forwardResList: List[ForwardRes],
               paramsList: List[(DenseMatrix[Double], DenseVector[Double])],
               hiddenLayers: List[Layer],
               outputLayer: Layer): List[BackwardRes]= {
    val yPredicted = forwardResList.last.yCurrent(::, 0)
    val numExamples = feature.rows

    val dYPredicted = -(label /:/ (yPredicted + pow(10.0, -9)) - (1.0 - label) /:/ (1.0 - yPredicted + pow(10.0, -9)))
    var dYCurrent = DenseMatrix.zeros[Double](numExamples, 1)
    dYCurrent(::, 0) := dYPredicted

    paramsList
      .zip(forwardResList)
      .zip(Nil.::(outputLayer).:::(hiddenLayers))
      .reverse
      .map{f =>
        val (w, b) = f._1._1
        val forwardRes = f._1._2
        val layer = f._2

        val backwardRes = layer.backward(dYCurrent, forwardRes, w, b)
        dYCurrent = backwardRes.dYPrevious

        backwardRes
      }
      .reverse
  }

}
