package estuary.model

import breeze.linalg.{DenseMatrix, DenseVector}
import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.Layer
import estuary.components.optimizer.{Distributed, Optimizer}
import estuary.components.regularizer.Regularizer
import org.apache.log4j.Logger

import scala.collection.mutable

class FullyConnectedNNModel(override val hiddenLayers: Seq[Layer],
                            override val outputLayer: Layer,
                            override val learningRate: Double,
                            override val iterationTime: Int,
                            override val regularizer: Option[Regularizer]) extends Model {
  override val logger: Logger = Logger.getLogger(this.getClass)

  var params: Seq[DenseMatrix[Double]] = _
  var costHistory: mutable.MutableList[Double] = _
  override var labelsMapping: Vector[Int] = _

  def init(inputDim: Int, outputDim: Int, initializer: WeightsInitializer): Seq[DenseMatrix[Double]] = {
    allLayers.foldLeft(inputDim) { case (previousDim, layer) => layer.setPreviousHiddenUnits(previousDim); layer.numHiddenUnits }
    allLayers.last.setNumHiddenUnits(outputDim)
    params = allLayers.map { layer => layer.init(initializer) }
    params
  }

  /** Fully functional method. */
  def trainFunc(feature: DenseMatrix[Double], label: DenseMatrix[Double], allLayers: Seq[Layer],
                opConfig: Model.OptimizationConfig, optimizer: Optimizer): Seq[DenseMatrix[Double]] = {
    optimizer.setIteration(opConfig.iterationTime).setLearningRate(opConfig.learningRate)
    optimizer match {
      case op: Distributed => op.parOptimize(feature, label, this, opConfig.initParams)
      case _ => optimizer.optimize(feature, label)(opConfig.initParams)(forward)(backward)
    }
  }

  def predict(feature: DenseMatrix[Double]): DenseVector[Int] = {
    val yHat = forward(feature)
    val deOneHottedYHat = Model.deOneHot(yHat)
    Model.convertMatrixToVector(deOneHottedYHat, labelsMapping)
  }

  def forward(feature: DenseMatrix[Double], params: Seq[DenseMatrix[Double]]): DenseMatrix[Double] = {
    allLayers.zip(params).par.foreach { case (layer, param) => layer.setParam(param) }
    allLayers.foldLeft(feature) { (yPrevious, layer) => layer.forward(yPrevious) }
  }

  def backward(label: DenseMatrix[Double], params: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = {
    allLayers.zip(params).par.foreach { case (layer, param) => layer.setParam(param) }
    allLayers.scanRight((label, DenseMatrix.zeros[Double](1, 1))) { case (layer, (dYCurrent, _)) =>
      layer.backward(dYCurrent, regularizer)
    }.init.map(_._2).toList
  }

  def copyStructure = new FullyConnectedNNModel(hiddenLayers.map(_.copyStructure), outputLayer.copyStructure, learningRate, iterationTime, regularizer)
}
