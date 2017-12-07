package estuary.components.support

import breeze.linalg.DenseMatrix
import estuary.components.layers._

trait CanExportParam[-From, To] {
  def export(from: From): To
}

object CanExportParam {
  implicit val anyLayerCanExportParam: CanExportParam[Layer[Any], Any] = (from) => {
    from match {
      case f: ClassicLayer => implicitly[CanExportParam[ClassicLayer, DenseMatrix[Double]]].export(f)
      case f: ConvLayer => implicitly[CanExportParam[ConvLayer, DenseMatrix[Double]]].export(f)
      case f: DropoutLayer => implicitly[CanExportParam[DropoutLayer, None.type]].export(f)
      case f: PoolingLayer => implicitly[CanExportParam[PoolingLayer, None.type]].export(f)
      case _ => throw new Exception(s"Unsupported layer of type ${from.getClass}")
    }
  }
}

