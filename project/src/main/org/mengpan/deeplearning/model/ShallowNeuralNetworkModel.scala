package org.mengpan.deeplearning.model
import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics._
import org.apache.log4j.Logger
import org.mengpan.deeplearning.utils.{ActivationUtils, GradientUtils, MyDict}

import scala.collection.mutable

/**
  * Created by mengpan on 2017/8/23.
  */
class ShallowNeuralNetworkModel extends Model {
  var logger: Logger = Logger.getLogger("ShallowNeuralNetworkModel")

  override var learningRate: Double = _
  override var iterationTime: Int = _
  override val costHistory: mutable.TreeMap[Int, Double] = new mutable.TreeMap[Int, Double]()

  var w1: DenseMatrix[Double] = _
  var w2: DenseVector[Double] = _
  var b1: DenseVector[Double] = _
  var b2: Double = _

  var numHiddenUnits: Int = _
  var hiddenActivationFunc: DenseMatrix[Double] => DenseMatrix[Double] = _

  case class Params(w1: DenseMatrix[Double], b1: DenseVector[Double],
                    w2: DenseVector[Double], b2: Double)

  case class ForwardRes(z1: DenseMatrix[Double], y1: DenseMatrix[Double],
                        z2: DenseVector[Double], y2: DenseVector[Double])

  case class BackwardRes(dw1: DenseMatrix[Double], db1: DenseVector[Double],
                         dw2: DenseVector[Double], db2: Double)

  def setNumHiddenUnits(numHiddenUnits: Int): this.type = {
    this.numHiddenUnits = numHiddenUnits
    this
  }

  def setHiddenActivationFunc(hiddenActivationFunc: Byte): this.type = {
    this.hiddenActivationFunc = ActivationUtils.getActivationFunc(hiddenActivationFunc)
    this
  }

  override def train(feature: DenseMatrix[Double], label: DenseVector[Double]): ShallowNeuralNetworkModel.this.type = {
    var (w1, b1, w2, b2) = initializeParameters(feature.cols)

    (0 until this.iterationTime).foreach{i =>

      val params = Params(w1, b1, w2, b2)
      val (cost, backwardRes) = propagate(feature, label, params)
      val dw1 = backwardRes.dw1
      val db1 = backwardRes.db1
      val dw2 = backwardRes.dw2
      val db2 = backwardRes.db2

      if (i % 100 == 0) {
        logger.info("Cost in " + i + "th time of iteration: " + cost)
      }
      costHistory.put(i, cost)

      var adjustedLearningRate = this.learningRate / ((i/500).toInt + 1.0)
      adjustedLearningRate =
        if (cost > costHistory.get(i-1).getOrElse[Double](10.0)) {
          logger.info("Cost increased. Learning rate reduced to 1/50")
          adjustedLearningRate/50.0
          }
        else adjustedLearningRate
      w1 :-= adjustedLearningRate*dw1
      b1 :-= adjustedLearningRate*db1
      w2 :-= adjustedLearningRate*dw2
      b2 -= adjustedLearningRate*db2
    }


    this.w1 = w1
    this.b1 = b1
    this.w2 = w2
    this.b2 = b2
    this
  }

  override def predict(feature: DenseMatrix[Double]): DenseVector[Double] = {
    val params = Params(this.w1, this.b1, this.w2, this.b2)
    forward(feature, params).y2.map{yHat =>
      if (yHat > 0.5) 1.0 else 0.0
    }
  }

  private def initializeParameters(featureSize: Int): (DenseMatrix[Double], DenseVector[Double],
    DenseVector[Double], Double) = {
    val w1 = DenseMatrix.rand[Double](featureSize, this.numHiddenUnits) * 0.01
    val b1 = DenseVector.zeros[Double](this.numHiddenUnits)
    val w2 = DenseVector.rand[Double](this.numHiddenUnits) * 0.01
    val b2 = 0.0
    (w1, b1, w2, b2)
  }

  private def propagate(feature: DenseMatrix[Double], label: DenseVector[Double],
                        params: Params): (Double, BackwardRes) = {
    val forwardRes = forward(feature, params)

    val cost = calCost(label, forwardRes)

    val backwardRes = backward(feature, label, forwardRes, params)

    logger.debug("the (dw1, db1) is " + backwardRes.dw1 + ", " + backwardRes.db1)
    logger.debug("the (dw2, db2) is " + backwardRes.dw2 + ", " + backwardRes.db2)
    (cost, backwardRes)
  }

  private def forward(feature: DenseMatrix[Double], params: Params): ForwardRes = {
    val w1 = params.w1
    val b1 = params.b1
    val w2 = params.w2
    val b2 = params.b2

    val z1 = feature * w1 + DenseVector.ones[Double](feature.rows) * b1.t
    val y1 = this.hiddenActivationFunc(z1)
    val z2 = y1 * w2 + b2
    val y2 = sigmoid(z2)

    ForwardRes(z1, y1, z2, y2)
  }

  private def calCost(label: DenseVector[Double], forwardRes: ForwardRes): Double = {
    val y2 = forwardRes.y2

    -((label.t) * log(y2 + pow(10.0, -9))
      + (1.0 - label).t * log(1.0 - y2 + pow(10.0, -9))) / label.length.toDouble
  }

  private def backward(feature: DenseMatrix[Double], label: DenseVector[Double],
                       forwardRes: ForwardRes, params: Params): BackwardRes = {
    val z1 = forwardRes.z1
    val y1 = forwardRes.y1
    val z2 = forwardRes.z2
    val y2 = forwardRes.y2
    val w1 = params.w1
    val b1 = params.b1
    val w2 = params.w2
    val b2 = params.b2
    val numExamples = feature.rows

    val dz2 = y2 - label
    val dw2 = (y1.t * dz2) / numExamples.toDouble
    val db2 = sum(dz2) / numExamples.toDouble
    val dy1 = dz2 * w2.t
    val dz1 = if (this.hiddenActivationFunc == tanh)
                dy1 *:* GradientUtils.tanhGrad(z1)
              else dy1 *:* GradientUtils.reluGrad(z1)
    val dw1 = feature.t * dz1 / numExamples.toDouble
    val db1 = dz1.t * DenseVector.ones[Double](numExamples) / numExamples.toDouble

    BackwardRes(dw1, db1, dw2, db2)
  }

}
