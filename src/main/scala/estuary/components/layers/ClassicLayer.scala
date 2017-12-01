package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.pow
import estuary.components.initializer.WeightsInitializer
import estuary.components.regularizer.Regularizer
import estuary.components.support.{CanAutoInit, CanBackward, CanForward, CanSetParam}
import ClassicLayer._
import estuary.model.Model.normalize

/**
  * Interface for neural network's layer.
  */
trait ClassicLayer extends Layer with Activator {

  /** ClassicLayer hyperparameters */
  val batchNorm: Boolean
  var previousHiddenUnits: Int = _

  def setPreviousHiddenUnits(numHiddenUnits: Int): this.type = {
    this.previousHiddenUnits = numHiddenUnits
    this
  }

  /** ClassicLayer parameters to be learned during training */
  var w: DenseMatrix[Double] = _
  var b: DenseVector[Double] = _
  var alpha: DenseVector[Double] = _
  var beta: DenseVector[Double] = _

  /** Cache processed data */
  var yPrevious: DenseMatrix[Double] = _
  var z: DenseMatrix[Double] = _
  var meanZ: DenseVector[Double] = _
  var stddevZ: DenseVector[Double] = _
  var currentMeanZ: DenseVector[Double] = _
  var currentStddevZ: DenseVector[Double] = _
  var zNorm: DenseMatrix[Double] = _
  var zDelta: DenseMatrix[Double] = _
  var y: DenseMatrix[Double] = _

  def forward(yPrevious: DenseMatrix[Double])(implicit op: CanForward[ClassicLayer, DenseMatrix[Double], DenseMatrix[Double]]): DenseMatrix[Double] =
    op.forward(yPrevious, this)

  def forwardForPrediction(yPrevious: DenseMatrix[Double])(implicit op: CanForward[ClassicLayer, DenseMatrix[Double], DenseMatrix[Double]]): DenseMatrix[Double] =
    op.forward(yPrevious, this)

  def backward(dYCurrent: DenseMatrix[Double], regularizer: Option[Regularizer])(implicit op: CanBackward[ClassicLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])]): (DenseMatrix[Double], DenseMatrix[Double]) =
    op.backward(dYCurrent, this, regularizer)

  def init(initializer: WeightsInitializer): DenseMatrix[Double] = {
    if (this.batchNorm) {
      val (w_, alpha_, beta_) = implicitly[CanAutoInit[WeightsInitializer, (Int, Int), (DenseMatrix[Double], DenseVector[Double], DenseVector[Double])]].init((previousHiddenUnits, numHiddenUnits), initializer)
      w = w_
      alpha = alpha_
      beta = beta_
      DenseMatrix.vertcat(this.w, this.alpha.toDenseMatrix, this.beta.toDenseMatrix)
    }
    else {
      val (w_, b_) = implicitly[CanAutoInit[WeightsInitializer, (Int, Int), (DenseMatrix[Double], DenseVector[Double])]].init((previousHiddenUnits, numHiddenUnits), initializer)
      w = w_
      b = b_
      DenseMatrix.vertcat(this.w, this.b.toDenseMatrix)
    }
  }

  def getReguCost(regularizer: Option[Regularizer]): Double = regularizer match {
    case Some(rg) => rg.getReguCost(w)
    case None => 0.0
  }

  def setParam(param: DenseMatrix[Double])(implicit op: CanSetParam[ClassicLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseVector[Double], DenseVector[Double])],
                                           op2: CanSetParam[ClassicLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseVector[Double])]): Unit =
    if (this.batchNorm)
      op.set(param, this)
    else
      op2.set(param, this)

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

  def normalizeGrad(dZNorm: DenseMatrix[Double], z: DenseMatrix[Double], meanZ: DenseVector[Double], stddevZ: DenseVector[Double]): DenseMatrix[Double] = {
    val oneVector = DenseVector.ones[Double](dZNorm.rows)
    val oneMat = DenseMatrix.ones[Double](dZNorm.rows, dZNorm.rows)
    val n = dZNorm.rows.toDouble

    (oneVector * pow(stddevZ + 1E-9, -1.0).t) / n *:* (dZNorm * n + (oneMat * dZNorm) - (z - oneVector * meanZ.t) *:* (oneVector * pow(stddevZ + 1E-9, -2.0).t) *:* (oneMat * (dZNorm *:* (z - oneVector * meanZ.t))))
  }

  def normalizeGradVec(dZNorm: DenseMatrix[Double], z: DenseMatrix[Double], meanZ: DenseVector[Double], stddevZ: DenseVector[Double]): DenseMatrix[Double] = {
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

object ClassicLayer {
  implicit val classicLayerCanAutoInitBN: CanAutoInit[WeightsInitializer, (Int, Int), (DenseMatrix[Double], DenseVector[Double], DenseVector[Double])] =
    (shape: (Int, Int), initializer: WeightsInitializer) => {
      val w = initializer.init(shape._1, shape._2)
      val alpha = DenseVector.zeros[Double](shape._2)
      val beta = DenseVector.zeros[Double](shape._2)
      (w, alpha, beta)
    }

  implicit val classicLayerCanAutoInitNonBN: CanAutoInit[WeightsInitializer, (Int, Int), (DenseMatrix[Double], DenseVector[Double])] =
    (shape: (Int, Int), initializer: WeightsInitializer) => {
      val w = initializer.init(shape._1, shape._2)
      val b = DenseVector.zeros[Double](shape._2)
      (w, b)
    }

  implicit val classicLayerCanBackward: CanBackward[ClassicLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])] =
    (input, by, regularizer) => {
      if (by.batchNorm) {
        val numExamples = input.rows
        val n = numExamples.toDouble
        val oneVector = DenseVector.ones[Double](numExamples)

        val dZDelta = input *:* by.activate(by.zDelta)
        val dZNorm = dZDelta *:* (oneVector * by.beta.t)
        val dAlpha = dZDelta.t * oneVector / n
        val dBeta = (dZDelta *:* by.zNorm).t * oneVector / n

        //Vector version
        val dZ = by.normalizeGradVec(dZNorm, by.z, by.currentMeanZ, by.currentStddevZ)

        //Matrix version (preffered, bug worse results than normalizeGradVec, why?)
        //        val dZ = by.normalizeGrad(dZNorm, z, currentMeanZ, currentStddevZ)

        val dWCurrent = regularizer match {
          case None => by.yPrevious.t * dZ / n
          case Some(regu) => by.yPrevious.t * dZ / n + regu.getReguCostGrad(by.w)
        }

        val dYPrevious = dZ * by.w.t

        val grads = DenseMatrix.vertcat(dWCurrent, dAlpha.toDenseMatrix, dBeta.toDenseMatrix)

        (dYPrevious, grads)
      } else {
        val numExamples = input.rows
        val n = numExamples.toDouble

        val dZ = input *:* by.activateGrad(by.z)
        val dWCurrent = regularizer match {
          case None => by.yPrevious.t * dZ / n
          case Some(regu) => by.yPrevious.t * dZ / n + regu.getReguCostGrad(by.w)
        }
        val dBCurrent = (DenseVector.ones[Double](numExamples).t * dZ).t / numExamples.toDouble
        val dYPrevious = dZ * by.w.t

        val grads = DenseMatrix.vertcat(dWCurrent, dBCurrent.toDenseMatrix)

        (dYPrevious, grads)
      }
    }

  implicit val classicLayerCanForward: CanForward[ClassicLayer, DenseMatrix[Double], DenseMatrix[Double]] =
    (input, by) => {
      if (!by.batchNorm) {
        by.yPrevious = input
        val numExamples = input.rows
        by.z = input * by.w + DenseVector.ones[Double](numExamples) * by.b.t
        by.y = by.activate(by.z)
        by.y
      } else {
        by.yPrevious = input
        val numExamples = input.rows
        val oneVector = DenseVector.ones[Double](numExamples)

        val z = input * by.w
        val (znorm, meanVec, stddevVec) = normalize(z)

        by.zNorm = znorm
        by.meanZ = if (by.meanZ == null) meanVec else 0.99 * by.meanZ + 0.01 * meanVec
        by.stddevZ = if (by.stddevZ == null) stddevVec else 0.99 * by.stddevZ + 0.01 * stddevVec
        by.currentMeanZ = meanVec
        by.currentStddevZ = stddevVec

        by.zDelta = by.zNorm *:* (oneVector * by.beta.t) + oneVector * by.alpha.t
        by.y = by.activate(by.zDelta)
        by.y
      }
    }

  implicit val classicLayerCanSetParamBN: CanSetParam[ClassicLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseVector[Double], DenseVector[Double])] =
    (from, foor) => {
      val w = from(0 to from.rows - 3, ::)
      val alpha = from(from.rows - 2, ::).t
      val beta = from(from.rows - 1, ::).t
      foor.w := w
      foor.alpha := alpha
      foor.beta := beta
      (w, alpha, beta)
    }

  implicit val classicLayerCanSetParamNonBN: CanSetParam[ClassicLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseVector[Double])] =
    (from, foor) => {
      val w = from(0 to from.rows - 2, ::)
      val b = from(from.rows - 1, ::).t
      foor.w := w
      foor.b := b
      (w, b)
    }


}
