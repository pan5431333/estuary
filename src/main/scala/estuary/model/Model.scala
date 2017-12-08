package estuary.model

import java.io.FileWriter

import breeze.linalg.{DenseMatrix, DenseVector, max, sum}
import breeze.numerics.log
import estuary.components.initializer.{HeInitializer, WeightsInitializer}
import estuary.components.layers.LayerLike.ForPrediction
import estuary.components.layers._
import estuary.components.optimizer.{AkkaParallelOptimizer, Optimizer, ParallelOptimizer}
import estuary.components.regularizer.Regularizer
import estuary.components.support._
import estuary.support.CanTrain
import shapeless.{HList, HNil}

import scala.collection.mutable.ArrayBuffer


class Model(val hiddenLayers: HList, val outputLayer: ClassicLayer)
  extends ModelLike[Model] {

  protected var params: Seq[DenseMatrix[Double]] = _
  protected var costHistory: ArrayBuffer[Double] = new ArrayBuffer[Double]()
  protected var inputDim: Int = _
  protected var outputDim: Int = _

  lazy val allLayers: HList = {
    var res: HList = HNil
    res = outputLayer :: res
    for (l <- hiddenLayers.reverse) {
      res = l :: res
    }
    res
  }

  def multiNodesParTrain(op: AkkaParallelOptimizer[Seq[DenseMatrix[Double]]]): this.type = {
    val trainedParams = op.parOptimize(repr)
    this.params = trainedParams.asInstanceOf[Seq[DenseMatrix[Double]]]
    this.costHistory = op.costHistory
    this
  }

  def init[Params](featureDim: Int, labelDim: Int, initializer: WeightsInitializer = HeInitializer): Unit = {
    this.inputDim = featureDim
    this.outputDim = labelDim
    init(initializer)
  }

  def forwardAndCalCost[Input, Output, Params](feature: Input, label: Output, params: Params): Double = {
    setParams(params.asInstanceOf[Seq[DenseMatrix[Double]]])
    val yHat: DenseMatrix[Double] = forward(feature.asInstanceOf[DenseMatrix[Double]])
    Model.calCost(label.asInstanceOf[DenseMatrix[Double]], yHat)
  }

  def backwardWithGivenParams[Input, Output, Params](label: Input, params: Params): Output = {
    setParams(params.asInstanceOf[Seq[DenseMatrix[Double]]])
    backward(label.asInstanceOf[DenseMatrix[Double]], None).asInstanceOf[Output]
  }

  def copyStructure: Model = {
    val newHidden = hiddenLayers.map()
    val newModel = new Model(hiddenLayers.map(_.copyStructure), outputLayer.copyStructure.asInstanceOf[ClassicLayer])
    newModel
  }
}


/** Util method for Neural Network Models */
object Model {

  def saveDenseMatricesToDisk(dms: Seq[DenseMatrix[_]], path: String): Unit = {
    val matrixSB = new StringBuilder()
    for (dm <- dms) {
      matrixSB.append(s"nRows: ${dm.rows}\n")
      matrixSB append s"nCols: ${dm.cols}\n"
      matrixSB append s"data: \n"
      for (d <- dm.data) {
        matrixSB append (d.toString + ",")
      }
      matrixSB append "\n\n"
    }
    val res = matrixSB.toString()
    val writer = new FileWriter(path)
    writer.write(res)
    writer.close()
  }

  def accuracy(label: DenseVector[Int], labelPredicted: DenseVector[Int]): Double = {
    val numCorrect = (0 until label.length).map { index =>
      if (label(index) == labelPredicted(index)) 1 else 0
    }.count(_ == 1)
    numCorrect.toDouble / label.length.toDouble
  }

  def evaluationTime[T](task: => T): Long = {
    val startTime = System.currentTimeMillis()
    task
    val endTime = System.currentTimeMillis()
    endTime - startTime
  }

  def deOneHot(yHat: DenseMatrix[Double]): DenseMatrix[Int] = {
    val deOneHottedMatrix = DenseMatrix.zeros[Int](yHat.rows, yHat.cols)
    for (i <- (0 until yHat.rows).par) {
      val sliced = yHat(i, ::)
      val maxRow = max(sliced)
      deOneHottedMatrix(i, ::) := sliced.t.map(index => if (index == maxRow) 1 else 0).t
    }
    deOneHottedMatrix
  }

  def calCost(label: DenseMatrix[Double], predicted: DenseMatrix[Double]): Double = {
    val originalCost = -sum(label *:* log(predicted + 1E-9)) / label.rows.toDouble
    originalCost
  }

  /**
    * Convert labels in a single vector to a matrix.
    * e.g. Vector(0, 1, 0, 1) => Matrix(Vector(1, 0, 1, 0), Vector(0, 1, 0, 1))
    * Vector(0, 1, 2) => Matrix(Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1))
    */
  def convertVectorToMatrix(labelVector: DenseVector[Int]): (DenseMatrix[Double], List[Int]) = {
    val labels = labelVector.toArray.toSet.toList.sorted //distinct elements by toSet.

    val numLabels = labels.size
    val res = DenseMatrix.zeros[Double](labelVector.length, numLabels)

    for ((label, i) <- labels.zipWithIndex.par) {
      val helperVector = DenseVector.ones[Int](labelVector.length) * label
      res(::, i) := elementWiseEqualCompare(labelVector, helperVector).map(_.toDouble)
    }
    (res, labels)
  }

  def convertMatrixToVector(labelMatrix: DenseMatrix[Int], labelsMapping: Vector[Int]): DenseVector[Int] = {
    val res = DenseVector.zeros[Int](labelMatrix.rows)
    for (i <- 0 until labelMatrix.cols) {
      res :+= labelMatrix(::, i) * labelsMapping(i)
    }
    res
  }

  def normalize(z: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = {
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

  /**
    * Compare two vector for equality in element-wise.
    * e.g. a = Vector(1, 2, 3), b = Vector(1, 0, 0), then return Vector(1, 0, 0)
    */
  def elementWiseEqualCompare(a: DenseVector[Int], b: DenseVector[Int]): DenseVector[Int] = {
    assert(a.length == b.length, "a.length != b.length")
    val compareArr = a.toArray.zip(b.toArray).par.map { case (i, j) =>
      if (i == j) 1 else 0
    }.toArray
    DenseVector(compareArr)
  }

  implicit val nnModelCanAutoInit: CanAutoInit[Model] =
    (foor: Model, initializer: WeightsInitializer) => {

      foor.outputLayer.setPreviousHiddenUnits(foor.hiddenLayers.last.numHiddenUnits)
      foor.hiddenLayers.foldLeft(foor.inputDim) {
        case (previousDim, layer: ClassicLayer) => layer.setPreviousHiddenUnits(previousDim); layer.numHiddenUnits
        case (_, layer) => layer.numHiddenUnits
      }

      foor.params = foor.allLayers
        .map { layer => layer.init(initializer); layer }
        .filter(_.hasParams)
        .map { layer => layer.getParam[DenseMatrix[Double]] }
    }

  implicit val nnModelCanSetParams: CanSetParam[Model, Seq[DenseMatrix[Double]]] =
    (from: Seq[DenseMatrix[Double]], foor: Model) => {

      foor.params = from
      foor.allLayers.filter(_.hasParams).zip(foor.params).par.foreach { case (layer, param) => layer.setParam(param) }
    }

  implicit val nnModelCanExportParams: CanExportParam[Model, Seq[DenseMatrix[Double]]] = {
    (from: Model) => from.params
  }

  implicit val nnModelCanForward: CanForward[Model, DenseMatrix[Double], DenseMatrix[Double]] =
    (input: DenseMatrix[Double], by: Model) => {
      by.setParams(by.params)
      by.allLayers.foldLeft(input) { case (yPrevious, layer) => layer.forward[DenseMatrix[Double], DenseMatrix[Double]](yPrevious) }
    }

  implicit val nnModelCanBackward: CanBackward[Model, DenseMatrix[Double], Seq[DenseMatrix[Double]]] =
    (input: DenseMatrix[Double], by: Model, regularizer: Option[Regularizer]) => {
      by.setParams(by.params)
      by.allLayers.scanRight[(DenseMatrix[Double], Option[DenseMatrix[Double]]), Seq[(DenseMatrix[Double], Option[DenseMatrix[Double]])]]((input, None)) { case (layer, (dYCurrent, _)) =>
        if (layer.hasParams) {
          val res = layer.backward[DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])](dYCurrent, regularizer)
          (res._1, Some(res._2))
        } else
          layer.backward[DenseMatrix[Double], (DenseMatrix[Double], None.type)](dYCurrent, regularizer)
      }.map(_._2).withFilter(_.isDefined).map(_.get).toList
    }

  implicit val nnModelCanForwardForPrediction: CanForward[Model, ForPrediction[DenseMatrix[Double]], DenseMatrix[Double]] =
    (input: ForPrediction[DenseMatrix[Double]], by: Model) => {
      val filtered = by.allLayers.filter(!_.isInstanceOf[DropoutLayer])
      filtered.foldLeft(input.input) { (yPrevious, layer) => layer.forward[DenseMatrix[Double], DenseMatrix[Double]](yPrevious) }
    }

  implicit val nnModelCanTrain: CanTrain[Model, DenseMatrix[Double], DenseMatrix[Double]] =
    (feature: DenseMatrix[Double], label: DenseMatrix[Double], optimizer: Optimizer, by: Model) => {
      val params = optimizer match {
        case op: ParallelOptimizer[Seq[DenseMatrix[Double]]] => op.parOptimize(feature, label, by, by.params)
        case op: AkkaParallelOptimizer[Seq[DenseMatrix[Double]]] => by.multiNodesParTrain(op).params
        case _ =>
          by.init(feature.cols, label.cols, HeInitializer)
          optimizer.optimize(feature, label)(by.params)(by.forwardAndCalCost)(by.backwardWithGivenParams)
      }

      by.params = params
      by.costHistory = optimizer.costHistory
      by.params
    }
}

