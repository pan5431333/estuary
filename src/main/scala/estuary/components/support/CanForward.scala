package estuary.components.support

import breeze.linalg.DenseMatrix
import estuary.components.layers._

trait CanForward[-By, -Input, +Output] {
  def forward(input: Input, by: By): Output
}

object CanForward {
//  implicit val anyLayerCanForward: CanForward[Layer, DenseMatrix[Double], DenseMatrix[Double]] =
//    (input, by) => {
//      by match {
//        case f: ClassicLayer => implicitly[CanForward[ClassicLayer, DenseMatrix[Double], DenseMatrix[Double]]].forward(input, f)
//        case f: ConvLayer => implicitly[CanForward[ConvLayer, DenseMatrix[Double], DenseMatrix[Double]]].forward(input, f)
//        case f: DropoutLayer => implicitly[CanForward[DropoutLayer, DenseMatrix[Double], DenseMatrix[Double]]].forward(input, f)
//        case f: PoolingLayer => implicitly[CanForward[PoolingLayer, DenseMatrix[Double], DenseMatrix[Double]]].forward(input, f)
//        case _ => throw new Exception(s"Unsupported layer of type ${by.getClass}")
//      }
//    }
}
