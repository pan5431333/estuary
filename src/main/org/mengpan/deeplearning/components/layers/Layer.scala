package org.mengpan.deeplearning.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.log4j.Logger
import org.mengpan.deeplearning.components.initializer.WeightsInitializer
import org.mengpan.deeplearning.components.regularizer.Regularizer

/**
  * Interface for neural network's layer.
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
  protected var yPrevious: DenseMatrix[Double] = _
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

  /**
    * Forward propagation of current layer.
    * @param yPrevious Output of previous layer, of the shape (n, d(l-1)), where
    *                  n: #training examples,
    *                  d(l-1): #hidden units in previous layer L-1.
    * @return Output of this layer, of the shape (n, d(l)), where
    *         n: #training examples,
    *         d(l): #hidden units in current layer L.
    */
  def forward(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {
    this.yPrevious = yPrevious
    y = this.batchNorm match {
      case true => forwardWithBatchNorm(yPrevious)
      case _ => forwardWithoutBatchNorm(yPrevious)
    }
    y
  }

  /**
    * Forward propagation of current layer for prediction's usage.
    * @note The difference between "forward" and "forwardForPrediction" is that,
    *       when the layer is batch normed, i.e. batchNorm is true, we use
    *       "forwardWithBatchNormForPrediction" instead of "forwardWithBatchNorm".
    * @param yPrevious Output of previous layer, of the shape (n, d(l-1)), where
    *                  n: #training examples,
    *                  d(l-1): #hidden units in previous layer L-1.
    * @return Output of this layer, of the shape (n, d(l)), where
    *         n: #training examples,
    *         d(l): #hidden units in current layer L.
    */
  def forwardForPrediction(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {
    this.batchNorm match {
      case true => forwardWithBatchNormForPrediction(yPrevious)
      case _ => forwardWithoutBatchNorm(yPrevious)
    }
  }

  /**
    * Backward propagation of current layer.
    * @param dYCurrent Gradients of current layer's output, DenseMatrix of shape (n, d(l))
    *                  where n: #training examples,
    *                  d(l): #hidden units in current layer L.
    * @return (dYPrevious, grads), where dYPrevious is gradients for output of previous
    *         layer; grads is gradients of current layer's parameters, i.e. for layer
    *         without batchNorm, parameters are w and b, for layers with batchNorm,
    *         parameters are w, alpha and beta.
    */
  def backward(dYCurrent: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    this.batchNorm match {
      case true => backWithBatchNorm(dYCurrent, yPrevious)
      case _ => backWithoutBatchNorm(dYCurrent, yPrevious)
    }
  }

  /**
    * Initialization for parameters in current layer.
    * @param initializer coule be HeInitializer, NormalInitializer of XaiverInitializer.
    * @return An DenseMatrix containing all parameters in current layer.
    *         For batchNorm is true, return's shape is (d(l-1) + 2, d(l)),
    *         For batchNorm is false, return's shape is (d(l-1) + 1, d(l))
    */
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

  /**
    * Get regularization cost, i.e. L1 norm or Frobinious norm of matrix w.
    * @param regularizer Could be L1Regularizer, or L2Regularizer.
    * @return Regularization cost of type Double.
    */
  def getReguCost(regularizer: Regularizer): Double = regularizer.getReguCost(w)


  /**
    * Set model parameters of current layer according to the input, which is a vertically
    * concatenated matrix containing all parameters.
    * @param param For batchNorm is true, param is of shape (d(l-1) + 2, d(l)),
    *              where d(l-1) is #hidden units in previous layer; d(l) is #hidden units
    *              in current layer. The top d(l-1) rows represent 'w', the (d(l-1)+1)th
    *              row represents transpose of 'alpha', the last row represents 'beta'.
    *              For batchNorm is false, param is of shape (d(l-1) + 1, d(l)). The top
    *              d(l-1) rows represent 'w', the last row represents 'b'.
    */
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
    meanZ = if (meanZ == null) meanVec else 0.9 * meanZ + 0.1 * meanVec
    stddevZ = if (stddevZ == null) stddevVec else 0.9 * stddevZ + 0.1 * stddevVec

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
