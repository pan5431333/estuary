package estuary.components.support

import breeze.linalg.DenseMatrix
import estuary.components.layers._
import estuary.components.regularizer.Regularizer

trait CanBackward[-By, -Input, +Output] {
  def backward(input: Input, by: By, regularizer: Option[Regularizer]): Output
}

object CanBackward {
//  implicit val anyLayerCanBackward: CanBackward[Layer, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])] =
//    (input, by, regularizer) => {
//      by match {
//        case f: SoftmaxLayer => implicitly[CanBackward[SoftmaxLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])]].backward(input, f, regularizer)
//        case f: ClassicLayer => implicitly[CanBackward[ClassicLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])]].backward(input, f, regularizer)
//        case f: ConvLayer => implicitly[CanBackward[ConvLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])]].backward(input, f, regularizer)
//        case f: DropoutLayer => implicitly[CanBackward[DropoutLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])]].backward(input, f, regularizer)
//        case f: PoolingLayer => implicitly[CanBackward[PoolingLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])]].backward(input, f, regularizer)
//        case _ => throw new Exception(s"Unsupported layer of type ${by.getClass}")
//      }
//    }
}
