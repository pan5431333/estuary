package org.mengpan.deeplearning.model

import breeze.linalg.{DenseMatrix, DenseVector, max}
import breeze.numerics.{abs, log, sigmoid}
import org.apache.log4j.Logger

import scala.collection.mutable

/**
  * Created by mengpan on 2017/8/15.
  */
class LogisticRegressionModel () extends Model {
  override val logger: Logger = Logger.getLogger("LogisticRegressionModel")

  var learningRate:Double = _
  var iterationTime: Int = _
  var w: DenseVector[Double] = _
  var b: Double = _

  @Override
  def train(feature: DenseMatrix[Double], label: DenseVector[Double]): this.type = {

    var (w, b) = initializeParams(feature.cols)

    (1 to this.iterationTime)
      .foreach{i =>
        val (cost, dw, db) = propagate(feature, label, w, b)

        if (i % 100 == 0)
          logger.info("Cost in " + i + "th time of iteration: " + cost)

        val adjustedLearningRate = this.learningRate / (log(i/1000 + 1) + 1)
        w :-= adjustedLearningRate * dw
        b -= adjustedLearningRate * db
      }


    this.w = w
    this.b = b
    this
  }

  @Override
  def predict(feature: DenseMatrix[Double]): DenseVector[Double] = {

    val yPredicted = sigmoid(feature * this.w + this.b).map{eachY =>
      if (eachY <= 0.05) 0.0 else 1.0
    }

    yPredicted
  }

  private def initializeParams(featureSize: Int): (DenseVector[Double], Double) = {
    val w = DenseVector.rand[Double](featureSize) * 0.01
    val b = 0.0
    (w, b)
  }

  private def propagate(feature: DenseMatrix[Double], label: DenseVector[Double], w: DenseVector[Double], b: Double): (Double, DenseVector[Double], Double) = {
    val numExamples = feature.rows
    val labelHat = sigmoid(feature * w + b)

    logger.debug("feature * w + b is " + feature * w + b)
    logger.debug("the feature's number of cols is " + feature.cols)
    logger.debug("the feature's number of rows is " + feature.rows)
    logger.debug("the labelHat is " + labelHat)

    val cost = -(label.t * log(labelHat) + (DenseVector.ones[Double](numExamples) - label).t * log(DenseVector.ones[Double](numExamples) - labelHat)) / numExamples

    val dw = feature.t * (labelHat - label) /:/ numExamples.toDouble
    val db = DenseVector.ones[Double](numExamples).t * (labelHat - label) / numExamples.toDouble

    logger.debug("the (dw, db) is " + dw + ", " + db)

    (cost, dw, db)
  }
}
