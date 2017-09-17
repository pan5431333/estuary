package org.mengpan.deeplearning.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.log4j.Logger
import org.mengpan.deeplearning.components.initializer.WeightsInitializer
import org.mengpan.deeplearning.components.regularizer.Regularizer

/**
  * Created by mengpan on 2017/8/26.
  */
trait Layer{
  private val logger = Logger.getLogger("Layer")

  /**Abstract Methods to be implemented*/
  protected def activationFuncEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double]
  protected def activationGradEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double]

  /**Layer parameters to be learned during training*/
  protected var w: DenseMatrix[Double] = _
  protected var b: DenseVector[Double] = _
  protected var alpha: DenseVector[Double] = _
  protected var beta: DenseVector[Double] = _

  /**Cache processed data*/
  protected var z: DenseMatrix[Double] = _
  protected var meanZ: DenseVector[Double] = _
  protected var stddevZ: DenseVector[Double] = _
  protected var zNorm: DenseMatrix[Double] = _
  protected var zDelta: DenseMatrix[Double] = _
  protected var y: DenseMatrix[Double] = _

  /**Layer hyperparameters and their setters*/
  var numHiddenUnits: Int = 0
  var batchNorm: Boolean = false
  var previousHiddenUnits: Int = _
  def setNumHiddenUnits(numHiddenUnits: Int): this.type = {
    assert(numHiddenUnits > 0, "Number of hidden units must be positive.")

    this.numHiddenUnits = numHiddenUnits
    this
  }
  def setBatchNorm(batchNorm: Boolean): this.type = {
    this.batchNorm = batchNorm
    this
  }
  def setPreviousHiddenUnits(numHiddenUnits: Int):this.type = {
    this.previousHiddenUnits = numHiddenUnits
    this
  }

  protected var yPrevious: DenseMatrix[Double] = _

  def forward(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {
    this.yPrevious = yPrevious
    y = this.batchNorm match {
      case true => forwardWithBatchNorm(yPrevious)
      case _ => forwardWithoutBatchNorm(yPrevious)
    }
    y
  }

  def forwardForPrediction(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {
    this.batchNorm match {
      case true => forwardWithBatchNormForPrediction(yPrevious)
      case _ => forwardWithoutBatchNorm(yPrevious)
    }
  }

  def backward(dYCurrent: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    this.batchNorm match {
      case true => backWithBatchNorm(dYCurrent, yPrevious)
      case _ => backWithoutBatchNorm(dYCurrent, yPrevious)
    }
  }

  def init(initializer: WeightsInitializer): DenseMatrix[Double] = {
    this.batchNorm match {
      case true => this.w = initializer.init(previousHiddenUnits, numHiddenUnits)
        this.alpha = DenseVector.zeros[Double](numHiddenUnits)
        this.beta = DenseVector.ones[Double](numHiddenUnits)
        DenseMatrix.vertcat(this.w, this.alpha.toDenseMatrix, this.beta.toDenseMatrix)
      case false => this.w = initializer.init(previousHiddenUnits, numHiddenUnits)
        this.b = DenseVector.zeros[Double](numHiddenUnits)
        DenseMatrix.vertcat(this.w, this.b.toDenseMatrix)
    }
  }

  def getReguCost(regularizer: Regularizer): Double = {
    this.batchNorm match {
      case true => regularizer.getReguCost(w)
      case _ => regularizer.getReguCost(w)
    }
  }

  def setParam(param: DenseMatrix[Double]): Unit = {
    if (this.batchNorm) {
      this.w = param(0 until previousHiddenUnits, ::)
      this.alpha = param(previousHiddenUnits, ::).t
      this.beta = param(param.rows-1, ::).t
    } else {
      this.w = param(0 until previousHiddenUnits, ::)
      this.b = param(param.rows - 1, ::).t
    }
  }

  private def forwardWithoutBatchNorm(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {

    val numExamples = yPrevious.rows
    z = yPrevious * w + DenseVector.ones[Double](numExamples) * b.t
    this.activationFuncEval(z)
  }

  private def forwardWithBatchNorm(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {
    val numExamples = yPrevious.rows
    val oneVector = DenseVector.ones[Double](numExamples)

    z = yPrevious * w
    val (znorm, meanVec, stddevVec) = normalize(z)

    zNorm = znorm
    meanZ = if (meanZ == null) meanVec else (meanZ + meanVec) / 2.0
    stddevZ = if (stddevZ == null) stddevVec else (stddevZ + stddevVec) / 2.0

    zDelta = (zNorm + oneVector * alpha.t) *:* (oneVector * beta.t)
    this.activationFuncEval(zDelta)
  }

  private def forwardWithBatchNormForPrediction(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {
    val numExamples = yPrevious.rows
    val oneVector = DenseVector.ones[Double](numExamples)

    z = yPrevious * w
    val zNorm = DenseMatrix.zeros[Double](z.rows, z.cols)

    for (j <- 0 until z.cols) {
      val jthCol = z(::, j)
      zNorm(::, j) := (jthCol - this.meanZ(j)) / (this.stddevZ(j) + 1E-9)
    }

    zDelta = (zNorm + oneVector * alpha.t) *:* (oneVector * beta.t)
    this.activationFuncEval(zDelta)
  }

  private def normalize(z: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = {
    val res = DenseMatrix.zeros[Double](z.rows, z.cols)
    val meanVec = DenseVector.zeros[Double](z.cols)
    val stddevVec = DenseVector.zeros[Double](z.cols)

    for (j <- 0 until z.cols) {
      val jthCol = z(::, j)
      val mean = breeze.stats.mean(jthCol)
      val stdDev = breeze.stats.stddev(jthCol)
      res(::, j) := (jthCol - mean) / (stdDev + 1E-9)
      meanVec(j) = mean
      stddevVec(j) = stdDev
    }

    (res, meanVec, stddevVec)
  }

  private def backWithoutBatchNorm(dYCurrent: DenseMatrix[Double], yPrevious: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {

    val numExamples = dYCurrent.rows

    val dZCurrent = dYCurrent *:* this.activationGradEval(z)
    val dWCurrent = yPrevious.t * dZCurrent / numExamples.toDouble
    val dBCurrent = (DenseVector.ones[Double](numExamples).t * dZCurrent).t / numExamples.toDouble
    val dYPrevious = dZCurrent * w.t

    val grads = DenseMatrix.vertcat(dWCurrent, dBCurrent.toDenseMatrix)

    (dYPrevious, grads)
  }

  private def backWithBatchNorm(dYCurrent: DenseMatrix[Double], yPrevious: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val numExamples = dYCurrent.rows
    val oneVector = DenseVector.ones[Double](numExamples)

    val dZDelta = dYCurrent *:* this.activationFuncEval(zDelta)
    val dZNorm = dZDelta *:* (oneVector * beta.t)
    val dAlpha = dZNorm.t * oneVector / numExamples.toDouble
    val dBeta = (dZDelta *:* (zNorm + oneVector * alpha.t)).t * oneVector / numExamples.toDouble
    val dZCurrent = dZNorm /:/ (oneVector * stddevZ.t)
    val dWCurrent = yPrevious.t * dZCurrent / numExamples.toDouble
    val dYPrevious = dZCurrent * w.t

    val grads = DenseMatrix.vertcat(dWCurrent, dAlpha.toDenseMatrix, dBeta.toDenseMatrix)

    (dYPrevious, grads)
  }

  override def toString: String = super.toString
}
