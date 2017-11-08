package estuary.model

import breeze.linalg.DenseMatrix
import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.{ClassicLayer, Layer}
import estuary.components.optimizer.Optimizer
import estuary.components.regularizer.Regularizer

import scala.collection.mutable.ArrayBuffer

class CNNModel(override val hiddenLayers: Seq[Layer],
               override val outputLayer: ClassicLayer,
               override val regularizer: Option[Regularizer])
  extends Model[Seq[DenseMatrix[Double]]]{

  var params: Seq[DenseMatrix[Double]] = _
  var costHistory: ArrayBuffer[Double] = _
  override var labelsMapping: Vector[Int] = _
  lazy val allLayers: Seq[Layer] = hiddenLayers :+ outputLayer

  override def init(inputDim: Int, outputDim: Int, initializer: WeightsInitializer) = ???

  override def predict(feature: DenseMatrix[Double]) = ???

  /** Fully functional method. */
  override def trainFunc(feature: DenseMatrix[Double], label: DenseMatrix[Double], allLayers: Seq[Layer], initParams: Seq[DenseMatrix[Double]], optimizer: Optimizer) = ???

  override def forward(feature: DenseMatrix[Double], params: Seq[DenseMatrix[Double]], allLayers: Seq[Layer]) = ???

  override def backward(label: DenseMatrix[Double], params: Seq[DenseMatrix[Double]]) = ???

  override def copyStructure = ???
}
