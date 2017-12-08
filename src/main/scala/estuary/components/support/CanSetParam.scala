package estuary.components.support

import breeze.linalg.DenseMatrix
import estuary.components.layers._

trait CanSetParam[For, From] {
  def set(from: From, foor: For): Unit
}

object CanSetParam {
//  implicit val anyLayerCanSetParam: CanSetParam[Layer, DenseMatrix[Double]] =
//    (from, foor) => {
//      foor match {
//        case f: ClassicLayer => implicitly[CanSetParam[ClassicLayer, DenseMatrix[Double]]].set(from, f)
//        case f: ConvLayer => implicitly[CanSetParam[ConvLayer, DenseMatrix[Double]]].set(from, f)
//        case f: DropoutLayer => implicitly[CanSetParam[DropoutLayer, None.type]].set(None, f)
//        case f: PoolingLayer => implicitly[CanSetParam[PoolingLayer, None.type]].set(None, f)
//        case _ => throw new Exception(s"Unsupported layer of type ${foor.getClass}")
//      }
//    }
}
