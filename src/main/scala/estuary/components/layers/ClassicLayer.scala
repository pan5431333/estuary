package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.pow
import estuary.components.initializer.WeightsInitializer
import estuary.components.regularizer.Regularizer

/**
  * Interface for neural network's layer.
  */
trait ClassicLayer extends Layer with Activator{
  /** ClassicLayer hyperparameters */
  protected val batchNorm: Boolean

  var previousHiddenUnits: Int = _

  def setPreviousHiddenUnits(numHiddenUnits: Int): this.type = {
    this.previousHiddenUnits = numHiddenUnits
    this
  }

  /** ClassicLayer parameters to be learned during training */
  protected var w: DenseMatrix[Double] = _
  protected var b: DenseVector[Double] = _
  protected var alpha: DenseVector[Double] = _
  protected var beta: DenseVector[Double] = _

  /** Cache processed data */
  protected var yPrevious: DenseMatrix[Double] = _
  protected var z: DenseMatrix[Double] = _
  protected var meanZ: DenseVector[Double] = _
  protected var stddevZ: DenseVector[Double] = _
  protected var currentMeanZ: DenseVector[Double] = _
  protected var currentStddevZ: DenseVector[Double] = _
  protected var zNorm: DenseMatrix[Double] = _
  protected var zDelta: DenseMatrix[Double] = _
  protected var y: DenseMatrix[Double] = _

  def isBatchNormed: Boolean = batchNorm

  /**
    * Forward propagation of current layer.
    *
    * @param yPrevious Output of previous layer, of the shape (n, d(l-1)), where
    *                  n: #training examples,
    *                  d(l-1): #hidden units in previous layer L-1.
    * @return Output of this layer, of the shape (n, d(l)), where
    *         n: #training examples,
    *         d(l): #hidden units in current layer L.
    */
  def forward(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {
    this.yPrevious = yPrevious
    y = if (this.batchNorm) forwardWithBatchNorm(yPrevious) else forwardWithoutBatchNorm(yPrevious)
    y
  }

  /**
    * Forward propagation of current layer for prediction's usage.
    *
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
    if (this.batchNorm) forwardWithBatchNormForPrediction(yPrevious)
    else forwardWithoutBatchNorm(yPrevious)
  }

  /**
    * Backward propagation of current layer.
    *
    * @param dYCurrent Gradients of current layer's output, DenseMatrix of shape (n, d(l))
    *                  where n: #training examples,
    *                  d(l): #hidden units in current layer L.
    * @return (dYPrevious, grads), where dYPrevious is gradients for output of previous
    *         layer; grads is gradients of current layer's parameters, i.e. for layer
    *         without batchNorm, parameters are w and b, for layers with batchNorm,
    *         parameters are w, alpha and beta.
    */
  def backward(dYCurrent: DenseMatrix[Double], regularizer: Option[Regularizer]): (DenseMatrix[Double], DenseMatrix[Double]) = {

    if (this.batchNorm) backWithBatchNorm(dYCurrent, yPrevious, regularizer)
    else backWithoutBatchNorm(dYCurrent, yPrevious, regularizer)
  }

  /**
    * Initialization for parameters in current layer.
    *
    * @param initializer coule be HeInitializer, NormalInitializer of XaiverInitializer.
    * @return An DenseMatrix containing all parameters in current layer.
    *         For batchNorm is true, return's shape is (d(l-1) + 2, d(l)),
    *         For batchNorm is false, return's shape is (d(l-1) + 1, d(l))
    */
  def init(initializer: WeightsInitializer): DenseMatrix[Double] = {
    if (this.batchNorm) {
      this.w = initializer.init(previousHiddenUnits, numHiddenUnits)
      this.alpha = DenseVector.zeros[Double](numHiddenUnits)
      this.beta = DenseVector.ones[Double](numHiddenUnits)
      DenseMatrix.vertcat(this.w, this.alpha.toDenseMatrix, this.beta.toDenseMatrix)
    }
    else {
      this.w = initializer.init(previousHiddenUnits, numHiddenUnits)
      this.b = DenseVector.zeros[Double](numHiddenUnits)
      DenseMatrix.vertcat(this.w, this.b.toDenseMatrix)
    }
  }

  /**
    * Get regularization cost, i.e. L1 norm or Frobinious norm of matrix w.
    *
    * @param regularizer Could be L1Regularizer, or L2Regularizer.
    * @return Regularization cost of type Double.
    */
  def getReguCost(regularizer: Option[Regularizer]): Double = regularizer match {
    case Some(rg) => rg.getReguCost(w)
    case None => 0.0
  }


  /**
    * Set model parameters of current layer according to the input, which is a vertically
    * concatenated matrix containing all parameters.
    *
    * @param param For batchNorm is true, param is of shape (d(l-1) + 2, d(l)),
    *              where d(l-1) is #hidden units in previous layer; d(l) is #hidden units
    *              in current layer. The top d(l-1) rows represent 'w', the (d(l-1)+1)th
    *              row represents transpose of 'alpha', the last row represents 'beta'.
    *              For batchNorm is false, param is of shape (d(l-1) + 1, d(l)). The top
    *              d(l-1) rows represent 'w', the last row represents 'b'.
    */
  def setParam(param: DenseMatrix[Double]): Unit = {
    if (this.batchNorm) {
      this.w = param(0 to param.rows - 3, ::)
      this.alpha = param(param.rows - 2, ::).t
      this.beta = param(param.rows - 1, ::).t
    } else {
      this.w = param(0 to param.rows - 2, ::)
      this.b = param(param.rows - 1, ::).t
    }
  }

  private def forwardWithoutBatchNorm(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {

    val numExamples = yPrevious.rows
    z = yPrevious * w + DenseVector.ones[Double](numExamples) * b.t
    this.activate(z)
  }

  private def forwardWithBatchNorm(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {
    val numExamples = yPrevious.rows
    val oneVector = DenseVector.ones[Double](numExamples)

    z = yPrevious * w
    val (znorm, meanVec, stddevVec) = normalize(z)

    zNorm = znorm
    meanZ = if (meanZ == null) meanVec else 0.99 * meanZ + 0.01 * meanVec
    stddevZ = if (stddevZ == null) stddevVec else 0.99 * stddevZ + 0.01 * stddevVec
    currentMeanZ = meanVec
    currentStddevZ = stddevVec

    zDelta = zNorm *:* (oneVector * beta.t) + oneVector * alpha.t
    activate(zDelta)
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

    zDelta = zNorm *:* (oneVector * beta.t) + oneVector * alpha.t
    activate(zDelta)
  }

  private def normalize(z: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = {
    val res = DenseMatrix.zeros[Double](z.rows, z.cols)
    val meanVec = DenseVector.zeros[Double](z.cols)
    val stddevVec = DenseVector.zeros[Double](z.cols)

    for (j <- (0 until z.cols).par) {
      val jthCol = z(::, j)
      val mean = breeze.stats.mean(jthCol)
      val variance = breeze.stats.variance(jthCol)
      val stdDev = breeze.numerics.sqrt(variance + 1E-9)
      res(::, j) := (jthCol - mean) / stdDev
      meanVec(j) = mean
      stddevVec(j) = stdDev
    }

    (res, meanVec, stddevVec)
  }

  private def backWithoutBatchNorm(dYCurrent: DenseMatrix[Double], yPrevious: DenseMatrix[Double], regularizer: Option[Regularizer]): (DenseMatrix[Double], DenseMatrix[Double]) = {

    val numExamples = dYCurrent.rows
    val n = numExamples.toDouble

    val dZ = dYCurrent *:* activateGrad(z)
    val dWCurrent = regularizer match {
      case None => yPrevious.t * dZ / n
      case Some(regu) => yPrevious.t * dZ / n + regu.getReguCostGrad(w)
    }
    val dBCurrent = (DenseVector.ones[Double](numExamples).t * dZ).t / numExamples.toDouble
    val dYPrevious = dZ * w.t

    val grads = DenseMatrix.vertcat(dWCurrent, dBCurrent.toDenseMatrix)

    (dYPrevious, grads)
  }

  private def backWithBatchNorm(dYCurrent: DenseMatrix[Double], yPrevious: DenseMatrix[Double], regularizer: Option[Regularizer]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val numExamples = dYCurrent.rows
    val n = numExamples.toDouble
    val oneVector = DenseVector.ones[Double](numExamples)

    val dZDelta = dYCurrent *:* activate(zDelta)
    val dZNorm = dZDelta *:* (oneVector * beta.t)
    val dAlpha = dZDelta.t * oneVector / n
    val dBeta = (dZDelta *:* zNorm).t * oneVector / n

    //Vector version
    val dZ = normalizeGradVec(dZNorm, z, currentMeanZ, currentStddevZ)

    //Matrix version (preffered, bug worse results than normalizeGradVec, why?)
    //        val dZ = normalizeGrad(dZNorm, z, currentMeanZ, currentStddevZ)

    val dWCurrent = regularizer match {
      case None => yPrevious.t * dZ / n
      case Some(regu) => yPrevious.t * dZ / n + regu.getReguCostGrad(w)
    }

    val dYPrevious = dZ * w.t

    val grads = DenseMatrix.vertcat(dWCurrent, dAlpha.toDenseMatrix, dBeta.toDenseMatrix)

    (dYPrevious, grads)
  }

  protected def normalizeGrad(dZNorm: DenseMatrix[Double], z: DenseMatrix[Double], meanZ: DenseVector[Double], stddevZ: DenseVector[Double]): DenseMatrix[Double] = {
    val oneVector = DenseVector.ones[Double](dZNorm.rows)
    val oneMat = DenseMatrix.ones[Double](dZNorm.rows, dZNorm.rows)
    val n = dZNorm.rows.toDouble

    (oneVector * pow(stddevZ + 1E-9, -1.0).t) / n *:* (dZNorm * n + (oneMat * dZNorm) - (z - oneVector * meanZ.t) *:* (oneVector * pow(stddevZ + 1E-9, -2.0).t) *:* (oneMat * (dZNorm *:* (z - oneVector * meanZ.t))))
  }

  protected def normalizeGradVec(dZNorm: DenseMatrix[Double], z: DenseMatrix[Double], meanZ: DenseVector[Double], stddevZ: DenseVector[Double]): DenseMatrix[Double] = {
    val n = z.rows.toDouble

    //Vectorized version
    val dZ = DenseMatrix.zeros[Double](z.rows, z.cols)
    for (j <- (0 until z.cols).par) {
      val dZNormJ = dZNorm(::, j)
      val dZJ = (DenseMatrix.eye[Double](dZNormJ.length) / stddevZ(j) - DenseMatrix.ones[Double](dZNormJ.length, dZNormJ.length) / (n * stddevZ(j)) - (z(::, j) - meanZ(j)) * (z(::, j) - meanZ(j)).t / (n * pow(stddevZ(j), 3.0))) * dZNormJ
      dZ(::, j) := dZJ
    }
    dZ
  }

  override def toString: String =
    s"""
       |ClassicLayer: ${getClass.getSimpleName},
       |Number of Hidden Units: $numHiddenUnits,
       |Is Batch Normed? $batchNorm,
       |Previous Number of Hidden Units? $previousHiddenUnits
    """.stripMargin
}
