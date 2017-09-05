package org.mengpan.deeplearning.components
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{log, pow}
import org.mengpan.deeplearning.layers.Layer
import org.mengpan.deeplearning.utils.ResultUtils
import org.mengpan.deeplearning.utils.ResultUtils.BackwardRes

/**
  * Created by mengpan on 2017/9/5.
  */
class L2Regularizer extends Regularizer{

  override protected def getReguCost(paramsList:
                                     List[(DenseMatrix[Double], DenseVector[Double])]):
  Double = {
    paramsList
    .map(_._1.data.map(pow(_, 2)).reduce(_+_))
    .reduce(_+_) / 2.0
  }

  override def backward(feature: DenseMatrix[Double],
                        label: DenseVector[Double],
                        forwardResList: List[ResultUtils.ForwardRes],
                        paramsList: List[(DenseMatrix[Double], DenseVector[Double])],
                        hiddenLayers: List[Layer],
                        outputLayer: Layer):
  List[ResultUtils.BackwardRes] = {
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

        new BackwardRes(backwardRes.dYPrevious,
          backwardRes.dWCurrent + this.lambda * w / numExamples.toDouble,
          backwardRes.dBCurrent)
      }
      .reverse
  }
}
