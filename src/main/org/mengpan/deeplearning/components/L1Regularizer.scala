package org.mengpan.deeplearning.components
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{abs, pow}
import org.mengpan.deeplearning.layers.Layer
import org.mengpan.deeplearning.utils.ResultUtils
import org.mengpan.deeplearning.utils.ResultUtils.BackwardRes

/**
  * Created by mengpan on 2017/9/5.
  */
class L1Regularizer extends Regularizer{

  override protected def getReguCost(paramsList:
                                     List[(DenseMatrix[Double], DenseVector[Double])]):
  Double = {
    paramsList
      .map(_._1.data.map(abs(_)).reduce(_+_))
      .reduce(_+_)
  }

  override def backward(feature: DenseMatrix[Double],
                        label: DenseVector[Double],
                        forwardResList: List[ResultUtils.ForwardRes],
                        paramsList: List[(DenseMatrix[Double], DenseVector[Double])],
                        hiddenLayers: List[Layer],
                        outputLayer: Layer):
  List[ResultUtils.BackwardRes] = {
    val numExamples = label.length
    val yHat = forwardResList.last.yCurrent(::, 0)

    //+ pow(10.0, -9)防止出现被除数为0，NaN的情况
    val dYL = -(label /:/ (yHat + pow(10.0, -9)) - (1.0 - label) /:/ (1.0 - yHat + pow(10.0, -9)))
    var dYCurrent = DenseMatrix.zeros[Double](feature.rows, 1)
    dYCurrent(::, 0) := dYL

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
          backwardRes.dWCurrent + this.lambda / numExamples.toDouble * sign(w),
          backwardRes.dBCurrent)
      }
      .reverse
  }

  private def sign(w: DenseMatrix[Double]): DenseMatrix[Double] = {
    w.map(e => if (e > 0) 1.0 else if (e < 0) -1.0 else 0.0)
  }
}
