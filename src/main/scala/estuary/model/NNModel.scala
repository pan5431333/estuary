package estuary.model

import breeze.linalg.DenseMatrix
import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.LayerLike.ForPrediction
import estuary.components.layers._
import estuary.components.optimizer.{AkkaParallelOptimizer, Optimizer, ParallelOptimizer}
import estuary.components.regularizer.Regularizer
import estuary.components.support.{CanAutoInit, CanBackward, CanForward, CanSetParam}
import estuary.support.CanTrain

class NNModel(val hiddenLayers: Seq[Layer[Any]],
              val outputLayer: ClassicLayer)
  extends Model[Seq[DenseMatrix[Double]]] with ModelLike[Seq[DenseMatrix[Double]], NNModel] {

  lazy val allLayers = hiddenLayers :+ outputLayer

  def multiNodesParTrain(op: AkkaParallelOptimizer[Seq[DenseMatrix[Double]]]): this.type = {
    val trainedParams = op.parOptimize(repr)
    this.params = trainedParams
    this.costHistory = op.costHistory
    this
  }

  override def forwardAndCalCost(feature: DenseMatrix[Double], label: DenseMatrix[Double], params: Seq[DenseMatrix[Double]]): Double = {
    setParams(params)
    val yHat = forward(feature)
    ModelLike.calCost(label, yHat)
  }

  override def backwardWithGivenParams(label: DenseMatrix[Double], params: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = {
    setParams(params)
    backward(label, None)
  }

  def copyStructure: NNModel = {
    val newModel = new NNModel(hiddenLayers.map(_.copyStructure), outputLayer.copyStructure.asInstanceOf[ClassicLayer])
    newModel
  }
}

object NNModel {
  implicit val nnModelCanAutoInit: CanAutoInit[NNModel] =
    (foor: NNModel, initializer: WeightsInitializer) => {
      foor.outputLayer.setPreviousHiddenUnits(foor.hiddenLayers.last.numHiddenUnits)
      foor.hiddenLayers.foldLeft(foor.inputDim) {
        case (previousDim, layer: ClassicLayer) => layer.setPreviousHiddenUnits(previousDim); layer.numHiddenUnits
        case (_, layer) => layer.numHiddenUnits
      }
      foor.params = foor.allLayers
        .map { layer => layer.init(initializer); layer.getParam[Any] }
        .map {
          case None => DenseMatrix.zeros[Double](1, 1)
          case a: DenseMatrix[Double] => a
          case _ => throw new Exception("Model initialization failed")
        }
    }

  implicit val nnModelCanSetParams: CanSetParam[NNModel, Seq[DenseMatrix[Double]]] =
    (from: Seq[DenseMatrix[Double]], foor: NNModel) => {
      foor.params = from
      foor.allLayers.filter(_.hasParams).zip(foor.params).par.foreach { case (layer, param) => layer.setParam(param) }
    }

  implicit val nnModelCanForward: CanForward[NNModel, DenseMatrix[Double], DenseMatrix[Double]] =
    (input: DenseMatrix[Double], by: NNModel) => {
      by.setParams(by.params)
      by.allLayers.foldLeft(input) { case (yPrevious, layer) => layer.forward(yPrevious) }
    }

  implicit val nnModelCanBackward: CanBackward[NNModel, DenseMatrix[Double], Seq[DenseMatrix[Double]]] =
    (input: DenseMatrix[Double], by: NNModel, regularizer: Option[Regularizer]) => {
      by.setParams(by.params)
      by.allLayers.scanRight((input, DenseMatrix.zeros[Double](1, 1))) { case (layer, (dYCurrent, _)) =>
        layer.backward[DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])](dYCurrent, regularizer)
      }.init.map(_._2).toList
    }

  implicit val nnModelCanForwardForPrediction: CanForward[NNModel, ForPrediction[DenseMatrix[Double]], DenseMatrix[Double]] =
    (input: ForPrediction[DenseMatrix[Double]], by: NNModel) => {
      val filtered = by.allLayers.filter(!_.isInstanceOf[DropoutLayer])
      filtered.foldLeft(input.input) { (yPrevious, layer) => layer.forward(yPrevious) }
    }

  implicit val nnModelCanTrain: CanTrain[NNModel, DenseMatrix[Double], DenseMatrix[Double], Seq[DenseMatrix[Double]]] =
    (feature: DenseMatrix[Double], label: DenseMatrix[Double], initParam: Seq[DenseMatrix[Double]], optimizer: Optimizer, by: NNModel) => {
      val params = optimizer match {
        case op: ParallelOptimizer[Seq[DenseMatrix[Double]]] => op.parOptimize(feature, label, by, by.params)
        case op: AkkaParallelOptimizer[Seq[DenseMatrix[Double]]] => by.multiNodesParTrain(op).params
        case _ =>
          by.init(feature.cols, label.cols)
          optimizer.optimize(feature, label)(by.params)(by.forwardAndCalCost)(by.backwardWithGivenParams)
      }

      by.params = params
      by.costHistory = optimizer.costHistory
      by.params
    }
}
